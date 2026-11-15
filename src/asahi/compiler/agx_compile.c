/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * Copyright 2016 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "c11/threads.h"
#include "asahi/clc/asahi_clc.h"
#include "asahi/isa/disasm.h"
#include "agx3_transcode.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_instr_set.h"
#include "compiler/nir/nir_range_analysis.h"
#include "util/bitset.h"
#include "util/glheader.h"
#include "util/half_float.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_nir_texture.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx3_matrix_core.h"
#include "agx_opcodes.h"
#include "agx3_pack.h"
#include "agx3_rt_core.h"
#include "agx3_vary_core.h"
#include "glsl_types.h"
#include "nir.h"
#include "util/os_misc.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

/* clang-format off */
static const struct debug_named_value agx_debug_options[] = {
   {"shaders",   AGX_DBG_SHADERS,	"Dump shaders in NIR and AIR"},
   {"binaries",  AGX_DBG_BINARIES, "Dump AGX3 binary records without NIR or AIR"},
   {"shaderdb",  AGX_DBG_SHADERDB,	"Print statistics"},
   {"verbose",   AGX_DBG_VERBOSE,	"Disassemble verbosely"},
   {"internal",  AGX_DBG_INTERNAL,	"Dump even internal shaders"},
   {"novalidate",AGX_DBG_NOVALIDATE,"Skip IR validation in debug builds"},
   {"noopt",     AGX_DBG_NOOPT,     "Disable backend optimizations"},
   {"wait",      AGX_DBG_WAIT,      "Wait after all async instructions"},
   {"nopreamble",AGX_DBG_NOPREAMBLE,"Do not use shader preambles"},
   {"demand",    AGX_DBG_DEMAND,    "Bound tightly to register demand"},
   {"nosched",   AGX_DBG_NOSCHED,   "Do not schedule the shader"},
   {"spill",     AGX_DBG_SPILL,     "Spill (almost) everything"},
   {"nopromote", AGX_DBG_NOPROMOTE, "Do not promote constants to uniforms"},
   {"latecov",   AGX_DBG_LATECOV,   "Keep G17P coverage updates at the end of the fragment"},
   {"refuse",    AGX_DBG_REFUSE,    "Name refused instructions, without dumping shaders"},
   DEBUG_NAMED_VALUE_END
};
/* clang-format on */

DEBUG_GET_ONCE_FLAGS_OPTION(agx_compiler_debug, "AGX_MESA_DEBUG",
                            agx_debug_options, 0)

int agx_compiler_debug = 0;

/*
 * Pad binary to a given alignment and return aligned offset into the binary.
 */
static unsigned
agx_pad_binary(struct util_dynarray *dyn, uint32_t align)
{
   if (dyn->size % align) {
      unsigned ngrow = align - (dyn->size % align);
      memset(util_dynarray_grow_bytes(dyn, ngrow, 1), 0, ngrow);
   }

   assert((dyn->size % align) == 0);
   return dyn->size;
}

uint64_t
agx_get_compiler_debug(void)
{
   return debug_get_option_agx_compiler_debug();
}

/* The last compile refusal of this thread. It lives in thread-specific storage
 * rather than a __thread array: libgallium and libvulkan_asahi use the
 * initial-exec TLS model, and a dlopen() of either only gets glibc's small
 * static TLS surplus, which an 896-byte array exhausts (EGL then fails to load
 * with "cannot allocate memory in static TLS block"). */
#define AGX_REFUSAL_SIZE 896
static tss_t agx_refusal_key;
static once_flag agx_refusal_once = ONCE_FLAG_INIT;

static void
agx_refusal_key_init(void)
{
   tss_create(&agx_refusal_key, free);
}

static char *
agx_refusal_buffer(bool create)
{
   call_once(&agx_refusal_once, agx_refusal_key_init);
   char *buf = tss_get(agx_refusal_key);
   if (!buf && create) {
      buf = calloc(1, AGX_REFUSAL_SIZE);
      if (buf && tss_set(agx_refusal_key, buf) != thrd_success) {
         free(buf);
         buf = NULL;
      }
   }
   return buf;
}

const char *
agx_last_compile_refusal(void)
{
   const char *buf = agx_refusal_buffer(false);
   return buf ? buf : "";
}

static void
agx_set_refusal(const agx_context *ctx, const char *what, unsigned line)
{
   char *buf = agx_refusal_buffer(true);
   if (!buf)
      return;
   snprintf(buf, AGX_REFUSAL_SIZE, "%s%s (%s:%u)", what,
            ctx ? (ctx->is_preamble ? " [constant program]" : " [main]") : "",
            __FILE__, line);
}

/* Fast-linked fragment programs blend the colour epilog inline. */
uint32_t
agx_get_fs_structure_flags(void)
{
   return AGX_FS_STRUCTURE_INLINE_BLEND;
}

static agx_index
agx_cached_preload(agx_context *ctx, unsigned base, enum agx_size size)
{
   if (agx_is_null(ctx->preloaded[base])) {
      agx_block *block = agx_start_block(ctx);
      agx_builder b = agx_init_builder(ctx, agx_before_block(block));
      ctx->preloaded[base] = agx_preload(&b, agx_register(base, size));
   }

   return ctx->preloaded[base];
}

static agx_index
agx_tess_coord_x(agx_builder *b)
{
   return agx_cached_preload(b->shader, 4, AGX_SIZE_32);
}

static agx_index
agx_tess_coord_y(agx_builder *b)
{
   return agx_cached_preload(b->shader, 6, AGX_SIZE_32);
}

static agx_index
agx_vertex_id_zero_base(agx_builder *b)
{
   return agx_cached_preload(b->shader, AGX_ABI_VIN_VERTEX_ID_ZERO_BASE,
                             AGX_SIZE_32);
}

static agx_index
agx_vertex_id(agx_builder *b)
{
   return agx_cached_preload(b->shader, AGX_ABI_VIN_VERTEX_ID, AGX_SIZE_32);
}

static agx_index
agx_instance_id(agx_builder *b)
{
   return agx_cached_preload(b->shader, AGX_ABI_VIN_INSTANCE_ID, AGX_SIZE_32);
}

#define VARYING_NUM_COMPONENTS (VARYING_SLOT_MAX * 4)

struct coefficient_info {
   BITSET_DECLARE(smooth, VARYING_NUM_COMPONENTS);
   BITSET_DECLARE(flat, VARYING_NUM_COMPONENTS);
   BITSET_DECLARE(noperspective, VARYING_NUM_COMPONENTS);
};

static BITSET_WORD *
bitset_for_interp(struct coefficient_info *info, enum glsl_interp_mode mode)
{
   /* clang-format off */
   switch (mode) {
   case INTERP_MODE_NONE:
   case INTERP_MODE_SMOOTH:         return info->smooth;
   case INTERP_MODE_NOPERSPECTIVE:  return info->noperspective;
   case INTERP_MODE_FLAT:           return info->flat;
   default:                         UNREACHABLE("invalid interp mode");
   }
   /* clang-format on */
}

static bool
gather_cf(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* First handle frag coord loads */
   struct coefficient_info *info = data;
   if (intr->intrinsic == nir_intrinsic_load_frag_coord_z) {
      BITSET_SET(info->noperspective, VARYING_SLOT_POS + 2);
      return false;
   }
   if (intr->intrinsic == nir_intrinsic_load_frag_coord_w) {
      BITSET_SET(info->noperspective, VARYING_SLOT_POS + 3);
      return false;
   }

   /* Look for input loads and grab the instruction with the interp mode */
   nir_intrinsic_instr *bary;
   unsigned nr = 1;

   if (intr->intrinsic == nir_intrinsic_load_coefficients_agx) {
      bary = intr;
      /* Always load a scalar */
   } else if (intr->intrinsic == nir_intrinsic_load_interpolated_input) {
      bary = nir_src_as_intrinsic(intr->src[0]);
      nr = intr->num_components;

      /* Perspective interpolation internally reads W */
      if (nir_intrinsic_interp_mode(bary) != INTERP_MODE_NOPERSPECTIVE)
         BITSET_SET(info->noperspective, VARYING_SLOT_POS + 3);
   } else {
      return false;
   }

   BITSET_WORD *set = bitset_for_interp(data, nir_intrinsic_interp_mode(bary));
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_src *offset = nir_get_io_offset_src(intr);

   /* Mark the exact range for direct loads to minimize CF registers, but mark a
    * conservative bounding range for indirect array access.
    */
   if (nir_src_is_const(*offset)) {
      unsigned location = sem.location + nir_src_as_uint(*offset);
      unsigned start_comp = (location * 4) + nir_intrinsic_component(intr);

      BITSET_SET_COUNT(set, start_comp, nr);
   } else {
      unsigned start_comp = (sem.location * 4) + nir_intrinsic_component(intr);
      bool compact = sem.location == VARYING_SLOT_CLIP_DIST0 ||
                     sem.location == VARYING_SLOT_CLIP_DIST1;
      unsigned stride = compact ? 1 : 4;

      /* For now we have to assign CF for the whole vec4 to make indirect
       * indexiing work. This could be optimized later.
       */
      nr = stride;

      for (unsigned i = 0; i < sem.num_slots; ++i) {
         BITSET_SET_COUNT(set, start_comp + (i * stride), nr);
      }
   }

   return false;
}

/*
 * We assign all coefficient registers up front to ensure we have a consistent
 * layout required for indirects to work.
 */
static void
assign_coefficient_regs(nir_shader *nir, struct agx_varyings_fs *var)
{
   struct coefficient_info info = {0};
   nir_shader_intrinsics_pass(nir, gather_cf, nir_metadata_all, &info);

   /* W */
   if (BITSET_TEST(info.noperspective, VARYING_SLOT_POS + 3)) {
      var->bindings[var->nr_bindings++] = (struct agx_cf_binding){
         .cf_base = var->nr_cf++,
         .slot = VARYING_SLOT_POS,
         .offset = 3,
         .count = 1,
         .smooth = true,
      };
   }

   /* Z */
   if (BITSET_TEST(info.noperspective, VARYING_SLOT_POS + 2)) {
      var->bindings[var->nr_bindings++] = (struct agx_cf_binding){
         .cf_base = var->nr_cf++,
         .slot = VARYING_SLOT_POS,
         .offset = 2,
         .count = 1,
         .smooth = true,
      };

      var->reads_z = true;
   }

   static_assert(VARYING_SLOT_POS == 0, "special and handled first");

   for (unsigned i = VARYING_SLOT_POS + 1; i < VARYING_SLOT_MAX; ++i) {
      bool smooth = BITSET_TEST_COUNT(info.smooth, i * 4, 4);
      bool flat = BITSET_TEST_COUNT(info.flat, i * 4, 4);
      bool noperspective = BITSET_TEST_COUNT(info.noperspective, i * 4, 4);

      if (!(smooth || flat || noperspective))
         continue;

      /* From the GLSL 4.60 spec ("Input Layout Qualifiers"):
       *
       *    when location aliasing, the aliases sharing the location must have
       *    the same underlying numerical type and bit width (floating-point or
       *    integer, 32-bit versus 64-bit, etc.) and the same auxiliary storage
       *    and interpolation qualification.
       *
       * SPIR-V should obey this as well although the spec text is muddier.
       */
      assert((smooth + flat + noperspective) == 1 &&
             "slots must have consistent interpolation");

      BITSET_WORD *set = smooth ? info.smooth
                         : flat ? info.flat
                                : info.noperspective;

      /* Find the start offset */
      unsigned offset = 0;
      for (offset = 0; offset < 4 && !BITSET_TEST(set, (i * 4) + offset);
           ++offset)
         ;

      /* Find the end offset. TODO: Do we ever need to split into two bindings
       * to handle e.g. x_zw read masks?
       */
      unsigned count = 0;
      for (unsigned c = offset; c < 4; ++c) {
         if (BITSET_TEST(set, (i * 4) + c))
            count = c - offset + 1;
      }
      assert(count >= 1 && (count + offset) <= 4);

      var->bindings[var->nr_bindings++] = (struct agx_cf_binding){
         .cf_base = var->nr_cf,
         .slot = i,
         .offset = offset,
         .count = count,
         .smooth = !flat,
         .perspective = smooth,
      };

      var->nr_cf += count;
   }
}

static agx_index
agx_get_cf(agx_context *ctx, gl_varying_slot slot, unsigned offset)
{
   struct agx_varyings_fs *varyings = &ctx->out->varyings.fs;

   /* We already have an appropriate binding, find it */
   for (unsigned b = 0; b < varyings->nr_bindings; ++b) {
      if (varyings->bindings[b].slot == slot &&
          (slot != VARYING_SLOT_POS ||
           offset == varyings->bindings[b].offset)) {

         signed cf_offset = offset - varyings->bindings[b].offset;
         assert(cf_offset >= 0);

         return agx_immediate(varyings->bindings[b].cf_base + cf_offset);
      }
   }

   UNREACHABLE("all coefficient registers preassigned");
}

/* Builds a 64-bit hash table key for an index */
static uint64_t
agx_index_to_key(agx_index idx)
{
   STATIC_ASSERT(sizeof(idx) <= sizeof(uint64_t));

   uint64_t key = 0;
   memcpy(&key, &idx, sizeof(idx));
   return key;
}

/*
 * Extract a single channel out of a vector source. We split vectors with
 * p_split so we can use the split components directly, without emitting a
 * machine instruction. This has advantages of RA, as the split can usually be
 * optimized away.
 */
static agx_index
agx_emit_extract(agx_builder *b, agx_index vec, unsigned channel)
{
   agx_index *components = _mesa_hash_table_u64_search(b->shader->allocated_vec,
                                                       agx_index_to_key(vec));

   assert(components != NULL && "missing agx_emit_collect_to");

   return components[channel];
}

static agx_index
agx_extract_nir_src(agx_builder *b, nir_src src, unsigned channel)
{
   agx_index idx = agx_src_index(&src);

   /* We only deal with scalars, extract a single scalar if needed */
   if (nir_src_num_components(src) > 1)
      return agx_emit_extract(b, idx, channel);
   else
      return idx;
}

static void
agx_cache_collect(agx_builder *b, agx_index dst, unsigned nr_srcs,
                  agx_index *srcs)
{
   /* Lifetime of a hash table entry has to be at least as long as the table */
   agx_index *channels = ralloc_array(b->shader, agx_index, nr_srcs);

   for (unsigned i = 0; i < nr_srcs; ++i)
      channels[i] = srcs[i];

   _mesa_hash_table_u64_insert(b->shader->allocated_vec, agx_index_to_key(dst),
                               channels);
}

/*
 * Combine multiple scalars into a vector destination. This corresponds to
 * collect, lowered to moves (a shuffle in general) after register allocation.
 *
 * To optimize vector extractions, we record the individual channels
 */
static agx_instr *
agx_emit_collect_to(agx_builder *b, agx_index dst, unsigned nr_srcs,
                    agx_index *srcs)
{
   agx_cache_collect(b, dst, nr_srcs, srcs);

   if (nr_srcs == 1)
      return agx_mov_to(b, dst, srcs[0]);

   agx_instr *I = agx_collect_to(b, dst, nr_srcs);

   agx_foreach_src(I, s)
      I->src[s] = srcs[s];

   return I;
}

static agx_index
agx_emit_collect(agx_builder *b, unsigned nr_srcs, agx_index *srcs)
{
   agx_index dst = agx_vec_temp(b->shader, srcs[0].size, nr_srcs);
   agx_emit_collect_to(b, dst, nr_srcs, srcs);
   return dst;
}

static agx_index
agx_vec2(agx_builder *b, agx_index s0, agx_index s1)
{
   return agx_emit_collect(b, 2, (agx_index[]){s0, s1});
}

static agx_index
agx_pad_to_32(agx_builder *b, agx_index s)
{
   assert(s.size == AGX_SIZE_16);
   assert(agx_channels(s) == 1);

   agx_index srcs[2] = {s, agx_undef(AGX_SIZE_16)};
   agx_index dst = agx_vec_temp(b->shader, AGX_SIZE_32, 1);
   agx_emit_collect_to(b, dst, 2, srcs);
   return dst;
}

/* A 16-bit operand the AGX3B coverage-word path can use as a register. */
static agx_index
agx_g17p_reg16(agx_builder *b, agx_index x)
{
   assert(x.size == AGX_SIZE_16);
   return x.type == AGX_INDEX_NORMAL ? x : agx_mov(b, x);
}

static agx_index
agx_recollect_vector(agx_builder *b, nir_src vec)
{
   agx_index comps[NIR_MAX_VEC_COMPONENTS];
   unsigned nr = nir_src_num_components(vec);
   assert(nr <= ARRAY_SIZE(comps));

   for (unsigned i = 0; i < nr; ++i)
      comps[i] = agx_extract_nir_src(b, vec, i);

   return agx_emit_collect(b, nr, comps);
}

/*
 * Extract the lower or upper N-bits from a (2*N)-bit quantity. We use a split
 * without null destinations to let us CSE (and coalesce) the splits when both x
 * and y are split.
 */
static agx_instr *
agx_subdivide_to(agx_builder *b, agx_index dst, agx_index s0, unsigned comp)
{
   assert((s0.size == (dst.size + 1)) && "only 2x subdivide handled");
   assert((comp == 0 || comp == 1) && "too many components");

   /* Handle immediates specially so we don't have to constant fold splits. */
   if (s0.type == AGX_INDEX_IMMEDIATE) {
      unsigned bits = 16 * agx_size_align_16(dst.size);
      return agx_mov_imm_to(b, dst, (s0.value >> bits) & BITFIELD64_MASK(bits));
   }

   agx_instr *split = agx_split(b, 2, s0);
   split->dest[comp] = dst;
   split->dest[1 - comp] = agx_temp(b->shader, dst.size);
   return split;
}

void
agx_block_add_successor(agx_block *block, agx_block *successor)
{
   assert(block != NULL && successor != NULL);

   /* Cull impossible edges */
   if (block->unconditional_jumps)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(block->successors); ++i) {
      if (block->successors[i]) {
         if (block->successors[i] == successor)
            return;
         else
            continue;
      }

      block->successors[i] = successor;
      util_dynarray_append(&successor->predecessors, block);
      return;
   }

   UNREACHABLE("Too many successors");
}

/*
 * Splits an n-component vector (vec) into n scalar destinations (dests) using a
 * split pseudo-instruction.
 *
 * Pre-condition: dests is filled with agx_null().
 */
static void
agx_emit_split(agx_builder *b, agx_index *dests, agx_index vec, unsigned n)
{
   agx_instr *I = agx_split(b, n, vec);

   agx_foreach_dest(I, d) {
      dests[d] = agx_temp(b->shader, vec.size);
      I->dest[d] = dests[d];
   }
}

static void
agx_emit_cached_split(agx_builder *b, agx_index vec, unsigned n)
{
   agx_index dests[NIR_MAX_VEC_COMPONENTS];
   assert(n <= ARRAY_SIZE(dests));
   for (unsigned i = 0; i < n; ++i)
      dests[i] = agx_null();
   agx_emit_split(b, dests, vec, n);
   agx_cache_collect(b, vec, n, dests);
}

static void
agx_emit_load_const(agx_builder *b, nir_load_const_instr *instr)
{
   /* Ensure we've been scalarized and bit size lowered */
   unsigned bit_size = instr->def.bit_size;
   assert(instr->def.num_components == 1);

   /* Emit move, later passes can inline/push if useful */
   agx_mov_imm_to(b, agx_def_index(&instr->def),
                  nir_const_value_as_uint(instr->value[0], bit_size));
}

/*
 * Implement mul_high of 32-bit sources by doing a 32x32->64-bit multiply and
 * extracting only the high word.
 */
static agx_instr *
agx_mul_high_to(agx_builder *b, agx_index dst, agx_index P, agx_index Q,
                bool is_signed)
{
   assert(P.size == Q.size && "source sizes must match");
   assert(P.size == dst.size && "dest size must match");
   assert(P.size != AGX_SIZE_64 && "64x64 multiply should have been lowered");

   static_assert(AGX_SIZE_64 == (AGX_SIZE_32 + 1), "enum wrong");
   static_assert(AGX_SIZE_32 == (AGX_SIZE_16 + 1), "enum wrong");

   if (!is_signed) {
      P = agx_abs(P);
      Q = agx_abs(Q);
   }

   agx_index product = agx_temp(b->shader, P.size + 1);
   agx_imad_to(b, product, P, Q, agx_zero(), 0);

   return agx_subdivide_to(b, dst, product, 1);
}

static enum agx_format
agx_format_for_pipe(enum pipe_format format)
{
#define CASE(x)                                                                \
   if (format == (enum pipe_format)AIL_ISA_FORMAT_##x)                         \
      return AGX_FORMAT_##x;

   CASE(I8);
   CASE(I16);
   CASE(I32);
   CASE(F16);
   CASE(U8NORM);
   CASE(S8NORM);
   CASE(U16NORM);
   CASE(S16NORM);
   CASE(RGB10A2);
   CASE(SRGBA8);
   CASE(RG11B10F);
   CASE(RGB9E5);

#undef CASE
   UNREACHABLE("Invalid format");
}

static agx_index
cf_for_intrinsic(agx_builder *b, nir_intrinsic_instr *intr)
{
   /* Determine the base location, taking into account a constant offset */
   unsigned location = nir_intrinsic_io_semantics(intr).location;
   bool compact = location == VARYING_SLOT_CLIP_DIST0 ||
                  location == VARYING_SLOT_CLIP_DIST1;

   nir_src *offset = nir_get_io_offset_src(intr);
   if (nir_src_is_const(*offset)) {
      /* XXX: NIR is broken and uses constant offsets in slots but dynamic
       * offsets in scalars for compact varyings. This needs to be fixed
       * upstream.
       */
      location += nir_src_as_uint(*offset);
   }

   agx_index I = agx_get_cf(b->shader, location, nir_intrinsic_component(intr));

   /* If we have a non-constant offset, we add it to the CF. Offsets are in
    * vec4 slots (unless we're compact) but the CF is in components, so we need
    * to shift the offset by 2 before adding.
    */
   if (!nir_src_is_const(*offset)) {
      I = agx_iadd(b, I, agx_src_index(offset), compact ? 0 : 2);
   }

   return I;
}

static enum agx_interpolation
agx_interp_for_bary(nir_intrinsic_instr *bary, agx_index *sample_index)
{
   switch (bary->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
      return AGX_INTERPOLATION_CENTER;

   case nir_intrinsic_load_barycentric_centroid:
      return AGX_INTERPOLATION_CENTROID;

   case nir_intrinsic_load_barycentric_at_sample:
      *sample_index = agx_src_index(&bary->src[0]);
      return AGX_INTERPOLATION_SAMPLE;

   default:
      UNREACHABLE("should have been lowered");
   }
}

/* AGX3 removed AGX2's fused projective iterator. Build Apple's witnessed
 * scalar software sequence while values are still in SSA so register
 * allocation can place the temporary RHW, reciprocal, and channels safely.
 * G17P explicit at-sample interpolation first computes one ITRCOORDS token
 * from the low half of the sample index, then shares it across RHW and every
 * scalar channel. G17P centroid interpolation computes the token from the
 * coverage word instead. Dynamic coefficient indices remain on the legacy IR
 * path so the AGX3 packer rejects them instead of guessing. */
/* Pixel-centre 1/W is the same for every input of a fragment program, so
 * compute it once at the top, as Apple's compiler does, instead of once per
 * load in every block that interpolates. */
static agx_index
agx3_center_rcp_w(agx_context *ctx, agx_index rhw_cf)
{
   if (agx_is_null(ctx->agx3_center_rcp_w)) {
      agx_block *start = agx_start_block(ctx);
      agx_builder b = agx_init_builder(ctx, agx_after_block(start));

      /* Preloads stay first in the program. */
      agx_foreach_instr_in_block(start, I) {
         if (I->op != AGX_OPCODE_PRELOAD) {
            b.cursor = agx_before_instr(I);
            break;
         }
      }

      agx_index rhw = agx_temp(ctx, AGX_SIZE_32);
      agx_rhw_interp_agx3_to(&b, rhw, rhw_cf, agx_zero(),
                             AGX_INTERPOLATION_CENTER);
      ctx->agx3_center_rcp_w = agx_persp_rcp_agx3(&b, rhw, rhw_cf.value);
   }

   return ctx->agx3_center_rcp_w;
}

static bool
agx_emit_load_vary_agx3(agx_builder *b, agx_index dest, agx_index cf,
                        agx_index sample_index, unsigned components,
                        enum agx_interpolation interp, bool perspective,
                        nir_alu_type dest_type, unsigned read_mask)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2 ||
       (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
       cf.type != AGX_INDEX_IMMEDIATE)
      return false;

   agx_index interp_coords = agx_zero();
   if (interp == AGX_INTERPOLATION_SAMPLE) {
      /* Native G17G at-sample interpolation (lane-mesh/cwit-interp:
       * interpolate_at_sample with a constant, uniform or sample-ID index and
       * [[sample_{,no_}perspective]] at 1, 2 and 4 samples) has G17P's shape:
       * one ITRCOORDS token from the sample index, then the native projective
       * RHW/RCP/MUL sequence around token f32interp leaves. */
      if (sample_index.size != AGX_SIZE_16 &&
          sample_index.size != AGX_SIZE_32)
         return false;

      agx_index sample_half = sample_index;
      if (sample_index.size == AGX_SIZE_32) {
         sample_half = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, sample_half, sample_index, 0);
      }

      /* ITRCOORDS has only a register source. This also keeps a constant
       * sample ID out of generic immediate promotion later in the backend. */
      if (sample_half.type != AGX_INDEX_NORMAL)
         sample_half = agx_mov(b, sample_half);

      interp_coords = agx_temp(b->shader, AGX_SIZE_32);
      agx_itrcoords_agx3_to(b, interp_coords, sample_half, interp);
   } else if (interp == AGX_INTERPOLATION_CENTROID) {
      /* Apple's G17P and G17G centroid inputs, single-sample and multisample
       * alike, pass native W16 SR39 -- the covered samples -- straight to
       * ITRCOORDS. With one sample that is the pixel centre. G17G witnesses:
       * lane-mesh/cwit-interp iac/iac_np, cwit-interp2 cp/cnp at 1 and 4
       * samples ([[centroid_{,no_}perspective]], interpolate_at_centroid()). */
      agx_index coverage = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_sr_to(b, coverage, AGX_SR_INPUT_SAMPLE_MASK);

      interp_coords = agx_temp(b->shader, AGX_SIZE_32);
      agx_itrcoords_agx3_to(b, interp_coords, coverage, interp);
   } else if ((interp != AGX_INTERPOLATION_CENTER &&
               interp != AGX_INTERPOLATION_CENTROID) ||
              sample_index.type != AGX_INDEX_IMMEDIATE ||
              sample_index.value != 0) {
      return false;
   }

   /* G17P uses the native projective RHW/RCP/MUL sequence for centre, sample
    * and centroid inputs alike. G17G's Gallium fragment path keeps the
    * generic sequence: the native f32mul has no 16-bit destination form,
    * which fp16 varyings need, and the generic sequence is what the G17G GL
    * line was validated with. */
   const bool native_projective =
      b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
      (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       b->shader->key->dev.chip != AGX_CHIP_G17G);
   agx_index reciprocal_w = agx_null();
   if (perspective) {
      agx_index rhw_cf = agx_get_cf(b->shader, VARYING_SLOT_POS, 3);
      if (rhw_cf.type != AGX_INDEX_IMMEDIATE)
         return false;

      if (interp == AGX_INTERPOLATION_CENTER &&
          native_projective) {
         reciprocal_w = agx3_center_rcp_w(b->shader, rhw_cf);
      } else {
         agx_index rhw = agx_temp(b->shader, AGX_SIZE_32);
         if (native_projective)
            agx_rhw_interp_agx3_to(b, rhw, rhw_cf, interp_coords, interp);
         else
            agx_interp_agx3_to(b, rhw, rhw_cf, interp_coords, interp, true);
         reciprocal_w = native_projective
                           ? agx_persp_rcp_agx3(b, rhw, rhw_cf.value)
                           : agx_rcp(b, rhw);
      }
   }

   agx_index channels[4];
   for (unsigned c = 0; c < components; ++c) {
      if (!(read_mask & BITFIELD_BIT(c))) {
         channels[c] = agx_undef(dest.size);
         continue;
      }
      agx_index channel_cf = cf;
      channel_cf.value += c;

      agx_index raw = agx_temp(b->shader, AGX_SIZE_32);
      agx_interp_agx3_to(b, raw, channel_cf, interp_coords, interp,
                         perspective);
      if (perspective &&
          native_projective) {
         /* Native f32mul.ep4 identifies the coefficient involved in this
          * projective divide. Preserve it explicitly rather than trying to
          * reconstruct SSA provenance after physical register allocation. */
         channels[c] = agx_persp_mul_agx3(b, raw, reciprocal_w, channel_cf.value);
      } else {
         channels[c] = perspective ? agx_fmul(b, raw, reciprocal_w) : raw;
      }

      /* AGX3 interpolation is natively 32-bit. Vulkan's
       * storageInputOutput16 feature nevertheless permits a logical 16-bit
       * fragment input, so narrow only after interpolation. Vertex output
       * lowering performs the matching type-correct widening before ST_VARY.
       * This preserves integer bits for flat varyings and applies the required
       * floating-point conversion for smooth/noperspective float varyings. */
      if (dest.size == AGX_SIZE_16) {
         agx_index narrowed = agx_temp(b->shader, AGX_SIZE_16);
         if (nir_alu_type_get_base_type(dest_type) == nir_type_float)
            agx_fmov_to(b, narrowed, channels[c]);
         else
            agx_subdivide_to(b, narrowed, channels[c], 0);

         channels[c] = narrowed;
      }
   }

   agx_emit_collect_to(b, dest, components, channels);
   return true;
}

static void
agx_emit_load_vary(agx_builder *b, agx_index dest, nir_intrinsic_instr *instr)
{
   ASSERTED unsigned components = instr->num_components;
   nir_intrinsic_instr *bary = nir_src_as_intrinsic(instr->src[0]);

   assert(components >= 1 && components <= 4);

   agx_index sample_index = agx_zero();
   enum agx_interpolation interp = agx_interp_for_bary(bary, &sample_index);

   bool perspective =
      nir_intrinsic_interp_mode(bary) != INTERP_MODE_NOPERSPECTIVE;

   agx_index I = cf_for_intrinsic(b, instr);

   if (agx_emit_load_vary_agx3(b, dest, I, sample_index, components, interp,
                               perspective, nir_intrinsic_dest_type(instr),
                               nir_def_components_read(&instr->def))) {
      agx_emit_cached_split(b, dest, components);
      return;
   }

   /* For perspective interpolation, we project (multiply by 1/W) */
   if (perspective) {
      agx_index J = agx_get_cf(b->shader, VARYING_SLOT_POS, 3);
      agx_iterproj_to(b, dest, I, J, sample_index, components, interp);
   } else {
      agx_iter_to(b, dest, I, sample_index, components, interp);
   }

   agx_emit_cached_split(b, dest, components);
}

static void
agx_wait_pixel_mask(agx_builder *b, uint32_t mask)
{
   /* Background programs do not need to wait as they are the eldest pixels */
   if (b->shader->key->fs.ignore_tib_dependencies) {
      assert(b->shader->nir->info.internal);
      return;
   }

   /* No need to wait twice on a fence */
   mask &= ~b->shader->already_pixel_waited;
   if (mask == 0) {
      return;
   }

   agx_wait_pix(b, mask);

   /* Only mark the fence as waited if we're not in control flow. Eventually we
    * should do something smarter with a dataflow.
    */
   if (b->shader->total_nesting == 0) {
      b->shader->already_pixel_waited |= mask;
   }
}

/*
 * AGX3's `lstcol` carries only a width, so unlike AGX2's st_tile it cannot
 * normalize on the way out: an 8-bit normalized target has to arrive already
 * packed, two channels to a 16-bit register. `fltnorm` is that pack -- it
 * clamps and scales two 32-bit floats into the two halves of one 16-bit
 * register -- and Apple's own G17P fragment shaders spend exactly
 * ceil(channels / 2) of them before every 8-bit colour store.
 *
 * Returns the number of packed 16-bit registers, or 0 if this store is not one
 * this path can pack, in which case the caller keeps the unpacked value and
 * the AGX3 packer refuses the shader rather than storing raw floats through a
 * width that would truncate them.
 *
 * `srgb_lanes` names the channels (in `channels` order) that an sRGB target
 * encodes. The tile itself is the same U8NORM byte storage; Apple selects the
 * transfer function on `fltnorm` alone (agx3_fltnorm_core.h, "The sRGB
 * conversion modes"), and the packer reads the per-pack lane set from the
 * instruction's mask.
 */
static unsigned
agx_pack_normalized_agx3(agx_builder *b, const agx_index *channels,
                         unsigned nr_channels, enum agx_format format,
                         unsigned srgb_lanes, agx_index *packed)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
      return 0;

   if (format == AGX_FORMAT_RGB10A2) {
      if (nr_channels != 4)
         return 0;
      agx_index pack_sources[4];
      for (unsigned i = 0; i < 4; ++i) {
         if (channels[i].size == AGX_SIZE_32) {
            pack_sources[i] = channels[i];
         } else if (channels[i].size == AGX_SIZE_16) {
            /* As with the normalized 8-bit epilog below, a texture result
             * can arrive through the half export ABI. The native RGB10A2
             * pack consumes f32, so widen before packing instead of handing
             * an unpacked half vector to the tile-store encoder. */
            pack_sources[i] = agx_temp(b->shader, AGX_SIZE_32);
            agx_fmov_to(b, pack_sources[i], channels[i]);
         } else {
            return 0;
         }
      }
      packed[0] = agx_pack_rgb10a2_agx3(b, pack_sources[0], pack_sources[1],
                                      pack_sources[2], pack_sources[3]);
      return 1;
   }

   /* Apple packs an RGB9E5 or RG11B10 target's three float lanes with the
    * same native conversion (rgb9e5/native/rgb9e5.py, layouts 1 and 2). A
    * store of four channels, as a background clear writes, has no alpha lane
    * to keep. */
   if ((format == AGX_FORMAT_RGB9E5 || format == AGX_FORMAT_RG11B10F) &&
       agx_isa_is_agx3(b->shader->key->dev.encoding)) {
      if (nr_channels != 3 && nr_channels != 4)
         return 0;
      for (unsigned i = 0; i < 3; ++i)
         if (channels[i].size != AGX_SIZE_32)
            return 0;
      packed[0] = agx_temp(b->shader, AGX_SIZE_32);
      agx_pack_rgb10a2_agx3_to(b, packed[0], channels[0], channels[1],
                               channels[2], agx_null())
         ->format = format;
      return 1;
   }

   /* Both native encodings use U8NORM tile storage and select the sRGB
    * transfer function on FLTNORM, including a linear alpha lane. */
   if (format == AGX_FORMAT_SRGBA8) {
      format = AGX_FORMAT_U8NORM;
   } else {
      srgb_lanes = 0;
   }

   const bool norm8 =
      format == AGX_FORMAT_U8NORM || format == AGX_FORMAT_S8NORM;
   const bool norm16 =
      format == AGX_FORMAT_U16NORM || format == AGX_FORMAT_S16NORM;

   /* Native G17P and G17G use modes 2/3 for normalized 16-bit channels. G18
    * R/RG/RGBA16 constant, uniform and reload witnesses use the same
    * scalar/pair conversion through the G17G (agx3c) IR G18G compiles to. */
   if (!norm8 && !norm16)
      return 0;

   if (nr_channels < 1 || nr_channels > 4)
      return 0;

   /* Every fltnorm witness takes 32-bit float sources. A linked fragment
    * epilog may receive normalized colour through the 16-bit export ABI, so
    * reproduce Apple's native half-to-float conversion before packing. */
   agx_index pack_sources[4];
   for (unsigned i = 0; i < nr_channels; ++i) {
      if (channels[i].size == AGX_SIZE_32) {
         pack_sources[i] = channels[i];
      } else if (channels[i].size == AGX_SIZE_16) {
         pack_sources[i] = agx_temp(b->shader, AGX_SIZE_32);
         agx_fmov_to(b, pack_sources[i], channels[i]);
      } else {
         return 0;
      }
   }

   unsigned n = 0;
   for (unsigned i = 0; i < nr_channels; i += 2) {
      const unsigned lanes = MIN2(2u, nr_channels - i);
      const agx_index hi =
         (lanes == 2) ? pack_sources[i + 1] : agx_null();
      /* The destination is a 16-bit half, not the sources' 32 bits, so the
       * generated temp-allocating helper cannot be used here. */
      agx_index dest = agx_temp(b->shader, AGX_SIZE_16);
      if (norm16)
         dest.channels_m1 = lanes - 1;
      agx_instr *pack =
         agx_fltnorm_agx3_to(b, dest, pack_sources[i], hi, format);
      pack->mask = (srgb_lanes >> i) & BITFIELD_MASK(lanes);
      if (norm16) {
         agx_index halves[2] = {agx_null()};
         agx_emit_split(b, halves, dest, lanes);
         for (unsigned c = 0; c < lanes; ++c)
            packed[n++] = halves[c];
      } else {
         packed[n++] = dest;
      }
   }
   return n;
}

/* Pure 8-bit integer outputs have already been clamped and narrowed to 16-bit
 * NIR lanes. AGX3 tile traffic instead carries two adjacent bytes in each
 * 16-bit register, so explicitly reproduce that physical packing here. */
static unsigned
agx_pack_integer_8_agx3(agx_builder *b, const agx_index *channels,
                        unsigned nr_channels, enum agx_format format,
                        agx_index *packed)
{
   /* AGX3c and AGX3d pack the same way: metal-nt applegpu_g17g/g18g give an
    * RGBA8Sint output two channels per half -- bit_and 255, then the odd
    * channel shifted by 8 into the same half -- ahead of one lstcol of
    * mask 15 (g18ref-out/probe/m6-int8-tile i8_sint4, i8_uint4). */
   if ((b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3B &&
        b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3C) ||
       format != AGX_FORMAT_I8 || nr_channels < 1 || nr_channels > 4)
      return 0;

   for (unsigned i = 0; i < nr_channels; ++i)
      if (channels[i].size != AGX_SIZE_16)
         return 0;

   unsigned n = 0;
   for (unsigned i = 0; i < nr_channels; i += 2) {
      if (i + 1 == nr_channels) {
         packed[n++] = channels[i];
         continue;
      }

      agx_index upper = agx_pad_to_32(b, channels[i + 1]);
      agx_index shifted = agx_temp(b->shader, AGX_SIZE_32);
      agx_bfi_to(b, shifted, agx_immediate(0), upper,
                 agx_immediate(8), 0);
      agx_index lower = agx_and(b, channels[i], agx_immediate(0xff));
      agx_index pair = agx_temp(b->shader, AGX_SIZE_16);
      agx_bitop_to(b, pair, lower, shifted, 0xe /* OR */);
      packed[n++] = pair;
   }

   return n;
}

/* A multi-component W8 payload is byte-packed in the register file: two
 * adjacent components occupy the low/high bytes of one 16-bit register. NIR
 * keeps each 8-bit component in its own 16-bit or zero-extended 32-bit lane,
 * so unpack the raw halves before exposing them to the rest of the backend. */
static void
agx_unpack_integer_8_agx3(agx_builder *b, agx_index dest,
                          const agx_index *halfs, unsigned nr_channels)
{
   agx_index channels[4] = {agx_null()};
   agx_index shift = agx_mov_imm(b, 32, 8);
   for (unsigned i = 0; i < nr_channels; ++i) {
      agx_index wide = agx_pad_to_32(b, halfs[i / 2]);
      if (i & 1)
         wide = agx_ushr(b, wide, shift);

      agx_index narrow = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, narrow, wide, 0);
      channels[i] = agx_and(b, narrow, agx_immediate(0xff));
      if (dest.size == AGX_SIZE_32) {
         agx_index zext = agx_temp(b->shader, AGX_SIZE_32);
         agx_mov_to(b, zext, channels[i]);
         channels[i] = zext;
      }
   }

   agx_emit_collect_to(b, dest, nr_channels, channels);
}

static void
agx_unpack_integer_8_memory_agx3(agx_builder *b, agx_index dest,
                                 agx_index packed, unsigned nr_channels)
{
   const unsigned nr_halfs = DIV_ROUND_UP(nr_channels, 2);
   agx_index halfs[2] = {agx_null()};
   for (unsigned i = 0; i < nr_halfs; ++i) {
      halfs[i] = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, halfs[i], packed, i);
   }

   agx_unpack_integer_8_agx3(b, dest, halfs, nr_channels);
}

/* Native G18 at one sample updates coverage with MOVMSK and then stores
 * through the whole-pixel immediate. Prove that the nearest coverage update
 * in this block already applied the store's mask before using that form. */
static bool
agx_store_mask_already_applied(nir_intrinsic_instr *store)
{
   nir_src mask = store->src[1];
   for (nir_instr *prev = nir_instr_prev(&store->instr); prev;
        prev = nir_instr_prev(prev)) {
      if (prev->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *update = nir_instr_as_intrinsic(prev);
      if (update->intrinsic != nir_intrinsic_sample_mask_agx)
         continue;
      if (!nir_srcs_equal(update->src[1], mask))
         return false;

      nir_src target = update->src[0];
      if (nir_src_is_const(target))
         return nir_src_as_uint(target) & 1;
      nir_alu_instr *alu = nir_src_as_alu(target);
      if (!alu)
         return false;
      if (alu->op == nir_op_inot)
         return nir_srcs_equal(alu->src[0].src, mask);
      if (alu->op != nir_op_ixor)
         return false;
      for (unsigned i = 0; i < 2; ++i) {
         if (nir_srcs_equal(alu->src[i].src, mask) &&
             nir_src_is_const(alu->src[1 - i].src) &&
             (nir_src_as_uint(alu->src[1 - i].src) & 1))
            return true;
      }
      return false;
   }
   return false;
}

static agx_instr *
agx_emit_local_store_pixel(agx_builder *b, nir_intrinsic_instr *instr)
{
   bool explicit = nir_intrinsic_explicit_coord(instr);

   /* TODO: Reverse-engineer interactions with MRT */
   if (b->shader->stage == MESA_SHADER_FRAGMENT) {
      agx_wait_pixel_mask(b, 0xC);
   }

   /* Compact the registers according to the mask */
   agx_index compacted[4] = {agx_null()};

   unsigned compact_count = 0;
   u_foreach_bit(i, nir_intrinsic_write_mask(instr)) {
      compacted[compact_count++] = agx_extract_nir_src(b, instr->src[0], i);
   }

   enum agx_format format = agx_format_for_pipe(nir_intrinsic_format(instr));

   /* The colour channels of an sRGB target are encoded, its alpha (the
    * fourth channel) is linear. Name them in compacted order. */
   unsigned srgb_lanes = 0;
   {
      unsigned k = 0;
      u_foreach_bit(i, nir_intrinsic_write_mask(instr)) {
         if (i < 3)
            srgb_lanes |= BITFIELD_BIT(k);
         ++k;
      }
   }

   agx_index packed[4] = {agx_null()};
   unsigned nr_packed = agx_pack_normalized_agx3(
      b, compacted, compact_count, format, srgb_lanes, packed);

   /* The packed sRGB bytes are stored through the U8NORM width. */
   if (nr_packed && format == AGX_FORMAT_SRGBA8)
      format = AGX_FORMAT_U8NORM;

   if (nr_packed == 0)
      nr_packed = agx_pack_integer_8_agx3(
         b, compacted, compact_count, format, packed);

   /* Native AGX3 tile stores carry a lane width, with the f32-to-f16
    * conversion in the shader. Both encodings have the narrowing FMUL form.
    * G18 native F16 stores consume contiguous half registers, including
    * packed UNORM targets whose physical tile representation is F16. */
   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       format == AGX_FORMAT_F16 && nr_packed == 0) {
      nr_packed = compact_count;
      for (unsigned i = 0; i < compact_count; ++i) {
         if (compacted[i].size == AGX_SIZE_32) {
            packed[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_fmov_to(b, packed[i], compacted[i]);
         } else {
            packed[i] = compacted[i];
         }
      }
   }

   agx_index collected =
      nr_packed ? agx_emit_collect(b, nr_packed, packed)
                : agx_emit_collect(b, compact_count, compacted);
   agx_index coords = explicit ? agx_src_index(&instr->src[2]) : agx_null();

   b->shader->out->tag_write_disable = false;
   agx_index sample = agx_src_index(&instr->src[1]);
   if (b->shader->key->dev.emit_agx3d &&
       b->shader->key->fs.nr_samples == 1 && !explicit &&
       !nir_src_is_const(instr->src[1]) &&
       agx_store_mask_already_applied(instr))
      sample = agx_immediate(0xff);

   agx_instr *st = agx_st_tile(b, collected, sample,
                               coords, format,
                               (nr_packed && (format == AGX_FORMAT_RGB10A2 ||
                                              format == AGX_FORMAT_RGB9E5 ||
                                              format == AGX_FORMAT_RG11B10F))
                                  ? 1
                                  : nir_intrinsic_write_mask(instr),
                               nir_intrinsic_base(instr), explicit,
                               nir_intrinsic_target(instr));

   /* Apple's G18 4x fragment releases the colour class explicitly after its
    * masked store (pixrel f14 0x20c: class 12 plus the final-release bit, the
    * packer's g17p_final_release). Without it the 4x draw hung in every tile
    * it covered (run165); 1x relies on the implicit release at end. */
   if (b->shader->stage == MESA_SHADER_FRAGMENT &&
       !b->shader->key->fs.ignore_tib_dependencies &&
       agx_is_g18g(&b->shader->key->dev) &&
       (b->shader->key->fs.nr_samples == 2 || b->shader->key->fs.nr_samples == 4))
      agx_signal_pix(b, 0xC);

   return st;
}

/* Stage one G17P depth/stencil word under the class-1 transaction the first
 * write of the shader opens and the last releases. */
static agx_instr *
agx_emit_zs_word_agx3(agx_builder *b, nir_intrinsic_instr *instr,
                      agx_index z, agx_index s, unsigned base,
                      agx_index coverage, agx_index prepared_depth,
                      bool bracketed)
{
   const bool write_z = base & 1;
   const bool write_s = base & 2;

   agx_index word = agx_temp(b->shader, AGX_SIZE_32);
   if (write_s) {
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
         /* Native G17G [[stencil]] and depth+stencil exports put the
          * target mask above the S8 reference: constant 7 stages 0xff07
          * in the high half, while coverage occupies the low half. */
         s = agx_or(b, s, agx_mov_imm(b, 16, 0xff00));
      }
      agx_emit_collect_to(b, word, 2, (agx_index[]){coverage, s});
   } else if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
      /* Native G17G DEPTHF keeps the target mask in bits24..31,
       * independently of SR2's duplicated low/high coverage bytes. */
      agx_index target = agx_mov_imm(b, 16, 0xff00);
      agx_emit_collect_to(b, word, 2, (agx_index[]){coverage, target});
   } else
      agx_mov_to(b, word, coverage);

   if (write_z) {
      agx_index depth;
      if (!agx_is_null(prepared_depth)) {
         depth = prepared_depth;
      } else if (nir_src_is_const(instr->src[1])) {
         const float value = nir_src_as_float(instr->src[1]);
         depth = agx_mov_imm(b, 32, fui(CLAMP(value, 0.0f, 1.0f)));
      } else {
         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
            agx_index zero = agx_immediate_f(0.0f);
            agx_index one = agx_immediate_f(1.0f);
            agx_index nonnegative = agx_fcmpsel(b, z, zero, z, zero,
                                                AGX_FCOND_GTN);
            depth = agx_fcmpsel(b, nonnegative, one, nonnegative, one,
                                 AGX_FCOND_LTN);
         } else {
            depth = agx_temp(b->shader, AGX_SIZE_32);
            agx_fmov_to(b, depth, z)->saturate = true;
         }
      }
      word = agx_vec2(b, word, depth);
   }

   b->shader->out->writes_sample_mask = true;
   /* A covered-sample loop's transaction is opened and closed around the
    * loop by zs_transaction_agx. */
   if (bracketed)
      return agx_zs_emit_agx3(b, word, base);
   agx_wait_pixel_mask(b, 0x1);
   agx_instr *I = agx_zs_emit_agx3(b, word, base);
   assert(b->shader->zs_writes_left > 0);
   if (--b->shader->zs_writes_left == 0)
      agx_signal_pix(b, 0x1);
   return I;
}

/*
 * Bits a run-time depth/stencil sample mask can still set. Per-sample lowering
 * hands the backend expressions like `exported_mask & (1 << id)`, whose value
 * is unknown but whose reach is not: an AND can only clear bits. Anything else
 * may set every sample.
 */
static unsigned
agx_possible_zs_samples(nir_def *def)
{
   if (def->num_components != 1)
      return UINT_MAX;

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return nir_src_as_uint(nir_src_for_ssa(def));

   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (nir_op_infos[alu->op].num_inputs == 1 &&
          (alu->op == nir_op_mov || alu->op == nir_op_u2u16 ||
           alu->op == nir_op_u2u32))
         return agx_possible_zs_samples(alu->src[0].src.ssa);

      if (alu->op == nir_op_iand)
         return agx_possible_zs_samples(alu->src[0].src.ssa) &
                agx_possible_zs_samples(alu->src[1].src.ssa);
   }

   return UINT_MAX;
}

static agx_instr *
agx_emit_store_zs(agx_builder *b, nir_intrinsic_instr *instr)
{
   unsigned base = nir_intrinsic_base(instr);
   bool write_z = base & 1;
   bool write_s = base & 2;

   agx_index z = agx_src_index(&instr->src[1]);
   agx_index s = agx_src_index(&instr->src[2]);

   assert(!write_z || z.size == AGX_SIZE_32);
   assert(!write_s || s.size == AGX_SIZE_16);

   if (b->shader->key->dev.emit_agx3d &&
       (b->shader->key->fs.nr_samples == 2 || b->shader->key->fs.nr_samples == 4) &&
       write_z && !write_s) {
      /* Native315/326 bind four/two-sample G18 depth fragments:
       * each DEPTHF preserves SR39's upper byte, selects one covered sample
       * in its low byte, and follows that word with saturated float depth.
       * The exports share one class-1 transaction. Unlike G17P, the
       * G18 word is not formed by shifting the upper coverage nibble. */
      const unsigned sample_mask = BITFIELD_MASK(b->shader->key->fs.nr_samples);
      unsigned reach = agx_possible_zs_samples(instr->src[0].ssa) & sample_mask;
      if (!reach)
         reach = sample_mask; /* Empty coverage still balances the transaction. */

      agx_index keep = agx_temp(b->shader, AGX_SIZE_16);
      agx_mov_to(b, keep, agx_src_index(&instr->src[0]));
      agx_index depth;
      if (nir_src_is_const(instr->src[1])) {
         const float value = nir_src_as_float(instr->src[1]);
         depth = agx_mov_imm(b, 32, fui(CLAMP(value, 0.0f, 1.0f)));
      } else {
         depth = agx_temp(b->shader, AGX_SIZE_32);
         agx_fmov_to(b, depth, z)->saturate = true;
      }
      agx_index depth_lo = agx_temp(b->shader, AGX_SIZE_16);
      agx_index depth_hi = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, depth_lo, depth, 0);
      agx_subdivide_to(b, depth_hi, depth, 1);
      agx_index high = agx_mov_imm(b, 16, 0xff00);

      agx_instr *last = NULL;
      b->shader->zs_writes_left += util_bitcount(reach) - 1;
      u_foreach_bit(i, reach) {
         agx_index sr39 = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_coverage_to(b, sr39, AGX_SR_INPUT_SAMPLE_MASK);
         agx_index covered = agx_and(
            b, agx_and(b, sr39, keep), agx_immediate(BITFIELD_BIT(i)));
         agx_index upper = agx_and(b, sr39, agx_mov_imm(b, 16, 0xff00));
         agx_index low = agx_iadd(b, covered, upper, 0);
         agx_index word = agx_vec_temp(b->shader, AGX_SIZE_16, 4);
         agx_emit_collect_to(b, word, 4,
                            (agx_index[]){low, high, depth_lo, depth_hi});
         b->shader->out->writes_sample_mask = true;
         agx_wait_pixel_mask(b, 0x1);
         last = agx_zs_emit_agx3(b, word, base);
         assert(b->shader->zs_writes_left > 0);
         if (--b->shader->zs_writes_left == 0)
            agx_signal_pix(b, 0x1);
      }
      return last;
   }

   /* G17P depthf reads staged dwords: coverage at half 0 and float32 depth
    * at half 2. Native FSEL.ep35 publishes the clamped depth directly to
    * staging[2], followed by a separate coverage COPY to staging[0].
    * Do not narrow depth into the coverage word's high half. Build every
    * full-width value before RA so the allocator owns the payload.
    *
    * A stencil reference occupies that high half instead. Apple's G17P
    * [[stencil]] fragments copy the 16-bit reference into the upper half of
    * the register holding the coverage, then stage the whole dword; with
    * [[depth(any)]] as well, depth still follows at half 2.
    */
   /* G17G stages the same two dwords through depthf with a mode field
    * (agx3_zs_core.c). Apple's fragments (gpu/m5-fragment-zs-constant-
    * reference.metal, -depth-) fold the target-sample nibble read from SR 2
    * as sr2 + (sr2 << 8) in the low half of the first dword, put
    * 0xff00 | stencil in its high half (0xff5a / 0xff01 for constant
    * stencil 0x5a / 1, 0xff00 when only depth is written) and publish the
    * saturated fp32 depth as the second dword. Build the vector as halves
    * so the allocator keeps the word and the depth one contiguous pair.
    *
    * The applegpu_g18g builds of the same fragments (gpu/m5-fragment-zs-
    * constant-reference.metal, -depth-) issue the same SR 2 read, iadd32
    * folds, 0xff00 | stencil half, copy and a depthf with an identical field
    * vector; only the depth clamp is one fsel3f32 where G17G uses
    * fself32 + fseli32, which is instruction selection, not this IR. So the
    * protocol is G17-class; before, AGX3D fell through to the AGX2 path.
    * Taken on G18G only: the G17G family keeps agx3c's forms below. */
   if (agx_is_g18g(&b->shader->key->dev)) {
      agx_index sr2 = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_coverage_1x_agx3_to(b, sr2);
      agx_index shifted = agx_iadd(b, agx_immediate(0), sr2, 4);
      agx_index low = agx_iadd(b, sr2, shifted, 4);
      agx_index high;
      if (write_s) {
         agx_index masked = agx_and(b, s, agx_immediate(0xff));
         high = agx_iadd(b, masked, agx_mov_imm(b, 16, 0xff00), 0);
      } else {
         high = agx_mov_imm(b, 16, 0xff00);
      }

      agx_index word;
      if (write_z) {
         agx_index depth;
         if (nir_src_is_const(instr->src[1])) {
            const float value = nir_src_as_float(instr->src[1]);
            depth = agx_mov_imm(b, 32, fui(CLAMP(value, 0.0f, 1.0f)));
         } else {
            depth = agx_temp(b->shader, AGX_SIZE_32);
            agx_fmov_to(b, depth, z)->saturate = true;
         }
         agx_index depth_lo = agx_temp(b->shader, AGX_SIZE_16);
         agx_index depth_hi = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, depth_lo, depth, 0);
         agx_subdivide_to(b, depth_hi, depth, 1);
         agx_index parts[4] = {low, high, depth_lo, depth_hi};
         word = agx_vec_temp(b->shader, AGX_SIZE_16, 4);
         agx_emit_collect_to(b, word, 4, parts);
      } else {
         word = agx_vec2(b, low, high);
      }

      b->shader->out->writes_sample_mask = true;
      agx_wait_pixel_mask(b, 0x1);
      agx_instr *I = agx_zs_emit_agx3(b, word, base);
      agx_signal_pix(b, 0x1);
      return I;
   }

   const unsigned samples = MAX2(b->shader->key->fs.nr_samples, 1);
   const unsigned mask =
      nir_src_is_const(instr->src[0]) ? nir_src_as_uint(instr->src[0]) : 0;
   const bool dynamic_mask = !nir_src_is_const(instr->src[0]);
   const bool sample_words =
      samples > 1 &&
      (mask == 0xff || util_bitcount(mask & BITFIELD_MASK(samples)) == 1);
   if (base & AGX_ZS_COVERED_SAMPLE) {
      /* Native M5 multisample fragments writing only depth or only stencil
       * (native G17G compiler zs_depth_{2,4,8}x, zs_cdepth_4x,
       * zs_depth_only_4x, zs_stencil_{2,4}x) issue one DEPTHF in a loop over
       * the covered samples, rereading SR39 on each iteration: its API mask
       * byte above that iteration's one sample. The loop and its transaction
       * come from agx_nir_lower_m5_zs_sample_loop. */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
          samples < 2 || (base & 3) == 3) {
         b->shader->unsupported_instruction = true;
         return NULL;
      }
      agx_index sr39 = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_sr_coverage_to(b, sr39, AGX_SR_INPUT_SAMPLE_MASK);
      agx_index covered = agx_and(b, sr39, agx_src_index(&instr->src[0]));
      agx_index api = agx_and(b, sr39, agx_mov_imm(b, 16, 0xff00));
      /* agx_nir_lower_m5_zs_sample_loop clamps the depth before the loop. */
      return agx_emit_zs_word_agx3(b, instr, z, s, base & 7,
                                  agx_or(b, api, covered),
                                  write_z ? z : agx_null(), true);
   }
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       samples > 1 && write_z && write_s && !dynamic_mask &&
       (mask & BITFIELD_MASK(samples)) == BITFIELD_MASK(samples)) {
      /* Native M5 whole-pixel depth and stencil (native G17G compiler
       * zs_depth_stencil_{2,4}x) is one DEPTHF naming every covered sample,
       * where depth or stencil alone loops over them. */
      agx_index sr39 = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_sr_coverage_to(b, sr39, AGX_SR_INPUT_SAMPLE_MASK);
      agx_index covered =
         agx_and(b, sr39, agx_immediate(BITFIELD_MASK(samples)));
      agx_index api = agx_and(b, sr39, agx_mov_imm(b, 16, 0xff00));
      return agx_emit_zs_word_agx3(b, instr, z, s, base,
                                  agx_or(b, api, covered), agx_null(), false);
   }
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       samples > 1) {
      /* G17G keeps SR39's API mask in the high byte and the changing
       * coverage in the low byte. Native multisample DEPTHF loops preserve
       * the former and select one covered sample per payload. The target
       * mask and optional stencil reference are added by the shared word
       * builder, independently of this coverage pair. */
      unsigned reach = agx_possible_zs_samples(instr->src[0].ssa) &
                       BITFIELD_MASK(samples);
      if (!reach)
         reach = BITFIELD_MASK(samples);

      agx_index keep = dynamic_mask
         ? agx_and(b, agx_src_index(&instr->src[0]),
                      agx_immediate(BITFIELD_MASK(samples)))
         : agx_immediate(mask & BITFIELD_MASK(samples));
      agx_instr *last = NULL;
      b->shader->zs_writes_left += util_bitcount(reach) - 1;
      u_foreach_bit(i, reach) {
         agx_index sr39 = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_coverage_to(b, sr39, AGX_SR_INPUT_SAMPLE_MASK);
         agx_index covered = agx_and(b, sr39, agx_immediate(BITFIELD_BIT(i)));
         if (dynamic_mask)
            covered = agx_and(b, covered, keep);
         else if (!(mask & BITFIELD_BIT(i)))
            covered = agx_zero();
         agx_index api = agx_and(b, sr39, agx_mov_imm(b, 16, 0xff00));
         last = agx_emit_zs_word_agx3(b, instr, z, s, base,
                                     agx_or(b, api, covered), agx_null(), false);
      }
      return last;
   }
   if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
       samples > 1 && dynamic_mask) {
      /* A split late shader keeps its discard predicate in a register. Fold
       * that predicate into the same per-sample words a compile-time mask
       * produces below: one DEPTHF per sample, each naming that sample alone
       * in the coverage nibble. Collapsing them into a single word whose
       * coverage nibble names several samples is not a form Apple emits, and
       * on G17P only sample zero receives the depth it stages -- measured on
       * renderpasses.*.multisample.d32_sfloat.samples_4, where samples 1..3
       * kept the clear and sample 0 alone held the written value. A sample
       * the predicate excludes gets an empty coverage nibble, exactly as a
       * sample an earlier kill removed from SR39 does. */
      unsigned reach =
         agx_possible_zs_samples(instr->src[0].ssa) & BITFIELD_MASK(samples);

      /* A mask that provably reaches nothing still needs a word for the
       * transaction to stay balanced; every sample's coverage is empty, so
       * which samples it names does not matter. */
      if (!reach)
         reach = BITFIELD_MASK(samples);

      agx_index keep = agx_and(b, agx_src_index(&instr->src[0]),
                               agx_immediate(BITFIELD_MASK(samples)));
      agx_instr *last = NULL;
      b->shader->zs_writes_left += util_bitcount(reach) - 1;
      u_foreach_bit(i, reach) {
         agx_index sr39 = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_coverage_to(b, sr39, AGX_SR_INPUT_SAMPLE_MASK);
         agx_index upper = agx_ushr(b, sr39, agx_immediate(4));
         agx_index covered =
            agx_and(b, agx_and(b, sr39, keep),
                    agx_immediate(BITFIELD_BIT(i)));
         last = agx_emit_zs_word_agx3(
            b, instr, z, s, base,
            agx_iadd(b, agx_immediate(15),
                     agx_iadd(b, upper, covered, 4), 4),
            agx_null(), false);
      }
      return last;
   }
   if ((b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C && samples == 1)) &&
       !sample_words) {
      /* At several samples, only a mask known at compile time has a
       * witnessed word; any other reads the one-sample coverage, which the
       * packer refuses there. */
      agx_index sr2 = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_coverage_1x_agx3_to(b, sr2);
      agx_index coverage = b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C
         ? agx_iadd(b, sr2, agx_iadd(b, agx_zero(), sr2, 4), 4)
         : agx_iadd(b, agx_immediate(15), agx_iadd(b, sr2, sr2, 4), 4);
      return agx_emit_zs_word_agx3(b, instr, z, s, base, coverage, agx_null(),
                                   false);
   } else if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Apple's 2x/4x fragments writing [[depth(any)]] or [[stencil]] per
       * [[sample_id]] write one DEPTHF per sample, each word built from SR39:
       * the API mask nibble, above the covered samples ANDed with that
       * sample's bit. A whole-pixel depth or stencil alone is written the
       * same way for every sample, but a whole-pixel depth and stencil is
       * written once, with all covered samples (metal-nt -arch applegpu_g17p,
       * msaa/native/zexp2 fs_id_{2,4}x and fs_var_4x, msaa/native/zmrt/zsten
       * fs_id_s_4x, fs_s_4x and fs_ds_4x).
       */
      const bool whole = mask == 0xff;
      const bool combined = whole && write_z && write_s;
      agx_instr *last = NULL;
      if (whole && !combined)
         b->shader->zs_writes_left += samples - 1;
      for (unsigned i = 0; i < (combined ? 1 : samples); ++i) {
         if (!whole && !(mask & BITFIELD_BIT(i)))
            continue;

         agx_index sr39 = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_coverage_to(b, sr39, AGX_SR_INPUT_SAMPLE_MASK);
         agx_index upper = agx_ushr(b, sr39, agx_immediate(4));
         agx_index covered = agx_and(
            b, sr39,
            agx_immediate(combined ? BITFIELD_MASK(4) : BITFIELD_BIT(i)));
         last = agx_emit_zs_word_agx3(
            b, instr, z, s, base,
            agx_iadd(b, agx_immediate(15),
                     agx_iadd(b, upper, covered, 4), 4),
            agx_null(), false);
      }
      return last;
   }

   if (write_z && write_s) {
      agx_index u2u32 = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, u2u32, s);
      s = u2u32;
   }

   agx_index zs = (write_z && write_s) ? agx_vec2(b, z, s) : write_z ? z : s;

   /* Not necessarily a sample mask but overlapping hw mechanism... Should
    * maybe rename this flag to something more general.
    */
   b->shader->out->writes_sample_mask = true;

   agx_wait_pixel_mask(b, 0x1);
   return agx_zs_emit(b, agx_src_index(&instr->src[0]), zs, base);
}

/* Unpack already-loaded normalized halves into `dest`. A byte format packs
 * two channels into each half; a 16-bit one occupies a half per channel and
 * is recollected into the pair the conversion consumes. Tile and device
 * traffic differ only in how the halves arrive, so both share this tail.
 *
 * `read_mask` names the channels anything reads: a pair with no read lane
 * needs no conversion. The channels are only cached as the collect's
 * sources, so readers take each `normflt` result where it was written and the
 * collected vector itself stays dead, like Apple's programs, which read the
 * conversion results in place. */
static void
agx_unpack_normalized_halves_agx3(agx_builder *b, agx_index dest,
                                  agx_index *halves, enum agx_format format,
                                  unsigned nr_comps, unsigned read_mask)
{
   /* An 8-bit sRGB target is U8NORM storage whose colour channels `normflt`
    * decodes (agx3_fltnorm_core.h, "The sRGB conversion modes"); the fourth
    * channel, alpha, is linear. The packer reads the lane set from the mask.
    * Witnessed on both native encodings. */
   unsigned srgb_lanes = 0;
   if (format == AGX_FORMAT_SRGBA8) {
      assert(b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2);
      format = AGX_FORMAT_U8NORM;
      srgb_lanes = BITFIELD_MASK(MIN2(nr_comps, 3));
   }

   const bool norm16 =
      format == AGX_FORMAT_U16NORM || format == AGX_FORMAT_S16NORM;
   agx_index channels[4] = {agx_null()};
   const unsigned conversions = DIV_ROUND_UP(nr_comps, 2);
   for (unsigned i = 0; i < conversions; ++i) {
      const unsigned lanes = MIN2(2u, nr_comps - i * 2);
      if (!(read_mask & (BITFIELD_MASK(lanes) << (i * 2)))) {
         for (unsigned c = 0; c < lanes; ++c)
            channels[i * 2 + c] = agx_undef(dest.size);
         continue;
      }

      /* One unpack per packed register; each yields a 32-bit pair, which the
       * split below turns back into the two scalar channels. */
      agx_index pair = agx_temp(b->shader, AGX_SIZE_32);
      pair.channels_m1 = lanes - 1;
      agx_index source = halves[i];
      if (norm16) {
         source = agx_temp(b->shader, AGX_SIZE_16);
         source.channels_m1 = lanes - 1;
         agx_emit_collect_to(b, source, lanes, &halves[i * 2]);
      }
      agx_normflt_agx3_to(b, pair, source, format)->mask =
         (srgb_lanes >> (i * 2)) & BITFIELD_MASK(lanes);

      agx_index unpacked[2] = {agx_null()};
      agx_emit_split(b, unpacked, pair, lanes);
      for (unsigned c = 0; c < lanes; ++c) {
         if (dest.size == AGX_SIZE_16) {
            channels[i * 2 + c] = agx_temp(b->shader, AGX_SIZE_16);
            agx_fmov_to(b, channels[i * 2 + c], unpacked[c]);
         } else {
            channels[i * 2 + c] = unpacked[c];
         }
      }
   }

   /* agx_emit_collect_to() records `channels` as the components of `dest`,
    * so consumers read them directly and the collect is dead code. A cached
    * split would make the collect live and cost one move per channel (four
    * in every RGBA8 blend epilog); Apple's inline blend has no such moves
    * (native fragment 0x10c..0x1b4). */
   agx_emit_collect_to(b, dest, nr_comps, channels);
}

/*
 * The tilebuffer read half of the same story as agx_pack_normalized_agx3():
 * an 8-bit normalized attachment comes back packed, two channels to a 16-bit
 * register, and `normflt` is the unpack -- one 16-bit register in, two 32-bit
 * floats out. Apple's programmable-blending shaders spend exactly
 * ceil(channels / 2) of them straight after the tile load.
 *
 * Returns true if it handled the load.
 */
static bool
agx_unpack_normalized_agx3(agx_builder *b, agx_index dest,
                           nir_intrinsic_instr *instr, enum agx_format format,
                           unsigned nr_comps)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
      return false;
   if (format == AGX_FORMAT_RGB10A2) {
      if (nr_comps != 4 || dest.size != AGX_SIZE_32)
         return false;
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_ld_tile_to(b, packed, agx_src_index(&instr->src[0]), agx_null(),
                    format, 1, nir_intrinsic_base(instr), false,
                    nir_intrinsic_target(instr));
      agx_unpack_rgb10a2_agx3_to(b, dest, packed);
      agx_emit_cached_split(b, dest, 4);
      return true;
   }
   /* Apple reads an RGB9E5 or RG11B10 target's three float lanes back with
    * the same native conversion (rgb9e5/native/rgb9e5.py fs_read). Neither
    * format has alpha, so a four-component read sees 1.0 there; a narrower
    * read takes the leading lanes. */
   if ((format == AGX_FORMAT_RGB9E5 || format == AGX_FORMAT_RG11B10F) &&
       agx_isa_is_agx3(b->shader->key->dev.encoding)) {
      if (nr_comps < 1 || nr_comps > 4 || dest.size != AGX_SIZE_32 ||
          (format == AGX_FORMAT_RGB9E5 && nr_comps < 3))
         return false;
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_ld_tile_to(b, packed, agx_src_index(&instr->src[0]), agx_null(),
                    format, 1, nir_intrinsic_base(instr), false,
                    nir_intrinsic_target(instr));
      agx_index rgb = nr_comps == 3 ? dest : agx_vec_temp(b->shader,
                                                          AGX_SIZE_32, 3);
      agx_unpack_rgb10a2_agx3_to(b, rgb, packed)->format = format;
      if (nr_comps != 3) {
         agx_index channels[4] = {agx_null()};
         agx_emit_split(b, channels, rgb, 3);
         channels[3] = agx_mov_imm(b, 32, fui(1.0f));
         agx_emit_collect_to(b, dest, nr_comps, channels);
      }
      agx_emit_cached_split(b, dest, nr_comps);
      return true;
   }
   /* sRGB: a U8NORM tile read, decoded by normflt (see
    * agx_unpack_normalized_halves_agx3). */
   const bool srgb = format == AGX_FORMAT_SRGBA8 &&
                     b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
   const enum agx_format tile_format = srgb ? AGX_FORMAT_U8NORM : format;
   const bool norm8 =
      tile_format == AGX_FORMAT_U8NORM || tile_format == AGX_FORMAT_S8NORM;
   const bool norm16 =
      format == AGX_FORMAT_U16NORM || format == AGX_FORMAT_S16NORM;
   if (!norm8 &&
       !(norm16 && agx_isa_is_agx3(b->shader->key->dev.encoding)))
      return false;
   if (nr_comps < 1 || nr_comps > 4 ||
       (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32))
      return false;

   const unsigned loaded_channels =
      norm16 ? nr_comps : DIV_ROUND_UP(nr_comps, 2);

   agx_index packed = agx_temp(b->shader, AGX_SIZE_16);
   packed.channels_m1 = loaded_channels - 1;
   agx_ld_tile_to(b, packed, agx_src_index(&instr->src[0]), agx_null(),
                  tile_format, BITFIELD_MASK(nr_comps),
                  nir_intrinsic_base(instr), false,
                  nir_intrinsic_target(instr));

   agx_index halves[4] = {agx_null()};
   agx_emit_split(b, halves, packed, loaded_channels);

   agx_unpack_normalized_halves_agx3(b, dest, halves, format, nr_comps,
                                     BITFIELD_MASK(nr_comps));
   return true;
}

static void
agx_emit_local_load_pixel(agx_builder *b, agx_index dest,
                          nir_intrinsic_instr *instr)
{
   agx_wait_pixel_mask(b, 0x8);

   unsigned nr_comps = instr->def.num_components;
   const enum agx_format format =
      agx_format_for_pipe(nir_intrinsic_format(instr));

   if (agx_unpack_normalized_agx3(b, dest, instr, format, nr_comps))
      return;

   /* Apple's reads of 8-bit integer attachments load the bytes packed, two
    * per 16-bit register, and then shift and mask the channels out
    * (blend/native/tilemask.py uread_*). */
   if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
       format == AGX_FORMAT_I8) {
      const unsigned nr_halfs = DIV_ROUND_UP(nr_comps, 2);
      agx_index packed = agx_temp(b->shader, AGX_SIZE_16);
      packed.channels_m1 = nr_halfs - 1;
      agx_ld_tile_to(b, packed, agx_src_index(&instr->src[0]), agx_null(),
                     format, BITFIELD_MASK(nr_comps),
                     nir_intrinsic_base(instr), false,
                     nir_intrinsic_target(instr));

      agx_index halfs[2] = {agx_null()};
      agx_emit_split(b, halfs, packed, nr_halfs);
      agx_unpack_integer_8_agx3(b, dest, halfs, nr_comps);
      agx_emit_cached_split(b, dest, nr_comps);
      return;
   }

   agx_ld_tile_to(b, dest, agx_src_index(&instr->src[0]), agx_null(), format,
                  BITFIELD_MASK(nr_comps), nir_intrinsic_base(instr), false,
                  nir_intrinsic_target(instr));
   agx_emit_cached_split(b, dest, nr_comps);
}

static bool
nir_is_coherent(nir_intrinsic_instr *instr)
{
   return nir_intrinsic_access(instr) &
          (ACCESS_COHERENT | ACCESS_VOLATILE | ACCESS_ATOMIC);
}

static void
agx_emit_load(agx_builder *b, agx_index dest, nir_intrinsic_instr *instr)
{
   agx_index addr = agx_src_index(&instr->src[0]);
   agx_index offset = agx_src_index(&instr->src[1]);
   enum agx_format fmt = agx_format_for_pipe(nir_intrinsic_format(instr));
   unsigned shift = nir_intrinsic_base(instr);
   const bool index_bytes = shift & AGX_NIR_MEMORY_INDEX_BYTES;
   const bool unpack_w8_word = shift & AGX_NIR_MEMORY_UNPACK_W8_WORD;
   assert(!index_bytes || b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2);
   const unsigned packed_count = (shift & AGX_NIR_MEMORY_PACKED_COUNT_MASK) >>
                                 AGX_NIR_MEMORY_PACKED_COUNT_SHIFT;
   assert(packed_count <= 4);
   assert(!packed_count || b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2);
   const int16_t byte_offset =
      (int16_t)(shift >> AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);
   assert(!byte_offset ||
          b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C);
   shift &= ~(AGX_NIR_MEMORY_INDEX_BYTES | AGX_NIR_MEMORY_PACKED_COUNT_MASK |
              AGX_NIR_MEMORY_UNPACK_W8_WORD | AGX_NIR_MEMORY_BYTE_OFFSET_MASK);

   assert(shift <= 3);

   /* Zero-extend offset if we're not sign-extending */
   if (!nir_intrinsic_sign_extend(instr))
      offset = agx_abs(offset);

   const bool norm8 = fmt == AGX_FORMAT_U8NORM || fmt == AGX_FORMAT_S8NORM;
   const bool norm16 = fmt == AGX_FORMAT_U16NORM || fmt == AGX_FORMAT_S16NORM;
   if ((b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 && norm8) ||
       (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 && norm16)) {
      const unsigned nr = instr->def.num_components;
      assert(nr >= 1 && nr <= 4 && packed_count == 0);
      assert(dest.size == AGX_SIZE_16 || dest.size == AGX_SIZE_32);

      /* Device memory provides raw bytes/halves, with no format conversion:
       * agx3_fmt_to_width() has no normalized device witness, so a load left
       * carrying one is refused at the packer and the draw is dropped. Keep
       * the element width and index scaling intact and reuse the grounded
       * normalized unpack the tile load already uses. G18's native Char3/
       * UChar3Normalized vertex fetches use W8/mask7 followed by pair/scalar
       * normflt, retaining the three-byte memory footprint. Its UNORM16 and
       * SNORM16 unpack uses the same scalar/pair operand footprint.
       * Byte loads reserve an aligned 32-bit payload even for a single
       * channel. */
      agx_index packed = norm8 ? agx_temp(b->shader, AGX_SIZE_32)
                              : agx_vec_temp(b->shader, AGX_SIZE_16, nr);
      agx_instr *load = agx_device_load_to(
         b, packed, addr, offset, norm8 ? AGX_FORMAT_I8 : AGX_FORMAT_I16,
         BITFIELD_MASK(nr), shift, nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_byte_offset = byte_offset;
      load->memory_is_packed = norm8 && nr > 1;

      agx_index halves[4] = {agx_null()};
      if (norm8) {
         for (unsigned i = 0; i < DIV_ROUND_UP(nr, 2); ++i) {
            halves[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_subdivide_to(b, halves[i], packed, i);
         }
      } else {
         agx_emit_split(b, halves, packed, nr);
      }
      /* Vertex fetch (agx_nir_lower_vbo, the only producer of normalized
       * constant loads) converts only the lanes the shader reads. */
      const bool vertex_fetch =
         instr->intrinsic == nir_intrinsic_load_constant_agx;
      agx_unpack_normalized_halves_agx3(
         b, dest, halves, fmt, nr,
         vertex_fetch ? nir_def_components_read(&instr->def)
                      : BITFIELD_MASK(nr));
      return;
   }

   /* AGX3 memory traffic is raw. Convert a normalized packed texel with
    * the same native instruction used by tile loads, after its real load. */
   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       fmt == AGX_FORMAT_RGB10A2) {
      assert(dest.size == AGX_SIZE_32 && instr->def.num_components == 4);
      assert(packed_count == 0);
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *load = agx_device_load_to(b, packed, addr, offset,
         AGX_FORMAT_I32, 1, shift, nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_byte_offset = byte_offset;
      agx_unpack_rgb10a2_agx3_to(b, dest, packed);
      agx_emit_cached_split(b, dest, 4);
      return;
   }

   /* The same one-word load and native unpack for R11G11B10_FLOAT: Apple's
    * G17P FloatRG11B10 vertex fetch is LDDEV W32 + `unpack` layout 2. */
   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       fmt == AGX_FORMAT_RG11B10F) {
      assert(dest.size == AGX_SIZE_32 && instr->def.num_components == 3);
      assert(packed_count == 0);
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *load = agx_device_load_to(b, packed, addr, offset,
         AGX_FORMAT_I32, 1, shift, nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_byte_offset = byte_offset;
      agx_unpack_rgb10a2_agx3_to(b, dest, packed)->format = fmt;
      agx_emit_cached_split(b, dest, 3);
      return;
   }

   /* Keep the original W8 transaction after folding the unsigned widenings.
    * Extract from its packed word so a useful operation can carry the wait.
    */
   if (unpack_w8_word) {
      assert(b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             b->shader->stage == MESA_SHADER_COMPUTE && !b->shader->is_preamble &&
             fmt == AGX_FORMAT_I8 && dest.size == AGX_SIZE_32 &&
             instr->def.num_components == 4 && packed_count == 0);
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *load = agx_device_load_to(b, packed, addr, offset, fmt,
                                           0xf, shift, nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_byte_offset = byte_offset;
      load->memory_is_packed = true;

      agx_index channels[4];
      for (unsigned i = 0; i < ARRAY_SIZE(channels); ++i) {
         agx_index shifted = i ? agx_ushr(b, packed, agx_immediate(8 * i))
                               : packed;
         channels[i] = i == 3 ? shifted
                              : agx_and(b, shifted, agx_immediate(0xff));
      }
      agx_emit_collect_to(b, dest, ARRAY_SIZE(channels), channels);
      return;
   }

   /* A scalar W8 load writes an aligned low half. Multi-component W8 loads
    * instead return a byte-packed payload, like G17P tile traffic. G18
    * returns vector bytes packed into one word; unpack them for both
    * halfword and 32-bit NIR lanes. */
   const bool is_g18 = agx_is_g18g(&b->shader->key->dev);
   if ((b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
       fmt == AGX_FORMAT_I8 &&
       (dest.size == AGX_SIZE_16 ||
        (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
         dest.size == AGX_SIZE_32)) && packed_count == 0) {
      const unsigned nr = instr->def.num_components;
      agx_index payload = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *load = agx_device_load_to(b, payload, addr, offset, fmt,
                                           BITFIELD_MASK(nr), shift,
                                           nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_byte_offset = byte_offset;

      if (nr == 1 && dest.size == AGX_SIZE_16) {
         agx_subdivide_to(b, dest, payload, 0);
      } else {
         /* G18G: a scalar byte widened to a 32-bit lane is not packed. */
         load->memory_is_packed = !is_g18 || nr > 1;
         agx_unpack_integer_8_memory_agx3(b, dest, payload, nr);
      }
      return;
   }

   if (is_g18 && fmt == AGX_FORMAT_I16 && dest.size == AGX_SIZE_32 &&
       packed_count == 0) {
      /* Raw W16 loads write adjacent halves, even when NIR requests
       * zero-extended 32-bit integer lanes. Load the physical halfword
       * vector first, then extend each component independently. */
      const unsigned nr = instr->def.num_components;
      assert(nr >= 1 && nr <= 4);
      agx_index payload = agx_vec_temp(b->shader, AGX_SIZE_16, nr);
      agx_instr *load = agx_device_load_to(b, payload, addr, offset, fmt,
                                           BITFIELD_MASK(nr), shift,
                                           nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_byte_offset = byte_offset;

      agx_index halves[4] = {agx_null()};
      agx_index channels[4] = {agx_null()};
      agx_emit_split(b, halves, payload, nr);
      for (unsigned i = 0; i < nr; ++i) {
         channels[i] = agx_temp(b->shader, AGX_SIZE_32);
         agx_mov_to(b, channels[i], halves[i]);
      }
      agx_emit_collect_to(b, dest, nr, channels);
      return;
   }

   agx_instr *load = agx_device_load_to(b, dest, addr, offset, fmt,
                      BITFIELD_MASK(packed_count ? packed_count : instr->def.num_components), shift,
                      nir_is_coherent(instr));
   load->memory_index_is_bytes = index_bytes;
      load->memory_byte_offset = byte_offset;
   load->memory_is_packed = packed_count != 0;
   /* nir_opt_access proved nothing writes this memory and it is not
    * volatile, so the read is a function of its address alone. The M5 spiller
    * may then repeat it in place of filling the value from a spill slot. */
   load->memory_can_reorder =
      b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
      nir_intrinsic_can_reorder(instr) && !nir_is_coherent(instr);
   agx_emit_cached_split(b, dest, instr->def.num_components);
}

static void
agx_emit_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index addr = agx_src_index(&instr->src[1]);
   agx_index offset = agx_src_index(&instr->src[2]);
   enum agx_format fmt = agx_format_for_pipe(nir_intrinsic_format(instr));
   unsigned shift = nir_intrinsic_base(instr);
   const bool index_bytes = shift & AGX_NIR_MEMORY_INDEX_BYTES;
   assert(!index_bytes || b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2);
   const int16_t byte_offset =
      (int16_t)(shift >> AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);
   assert(!byte_offset ||
          b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C);
   shift &= ~(AGX_NIR_MEMORY_INDEX_BYTES | AGX_NIR_MEMORY_BYTE_OFFSET_MASK);

   assert(shift <= 3);

   /* Zero-extend offset if we're not sign-extending */
   if (!nir_intrinsic_sign_extend(instr))
      offset = agx_abs(offset);

   agx_index value;
   unsigned mask;
   if ((b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
       fmt == AGX_FORMAT_RGB10A2) {
      assert(nir_src_num_components(instr->src[0]) == 4 &&
             nir_src_bit_size(instr->src[0]) == 32);
      value = agx_pack_rgb10a2_agx3(b,
         agx_extract_nir_src(b, instr->src[0], 0),
         agx_extract_nir_src(b, instr->src[0], 1),
         agx_extract_nir_src(b, instr->src[0], 2),
         agx_extract_nir_src(b, instr->src[0], 3));
      fmt = AGX_FORMAT_I32;
      mask = 1;
   } else if ((b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
               b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
              fmt == AGX_FORMAT_I8 &&
              nir_src_bit_size(instr->src[0]) == 8 &&
              nir_src_num_components(instr->src[0]) > 1) {
      /* Preserve one W8 vector transaction while packing its register payload
       * two bytes per 16-bit lane. */
      const unsigned nr = nir_src_num_components(instr->src[0]);
      agx_index channels[4] = {agx_null()};
      for (unsigned i = 0; i < nr; ++i)
         channels[i] = agx_extract_nir_src(b, instr->src[0], i);

      agx_index packed[2] = {agx_null()};
      const unsigned nr_packed =
         agx_pack_integer_8_agx3(b, channels, nr, fmt, packed);
      assert(nr_packed == DIV_ROUND_UP(nr, 2));
      value = agx_emit_collect(b, nr_packed, packed);
      mask = BITFIELD_MASK(nr);
   } else {
      value = agx_recollect_vector(b, instr->src[0]);
      mask = BITFIELD_MASK(nir_src_num_components(instr->src[0]));
   }
   agx_instr *store = agx_device_store(b, value, addr, offset, fmt, mask,
                                      shift, nir_is_coherent(instr));
   store->memory_index_is_bytes = index_bytes;
   store->memory_byte_offset = byte_offset;
   store->memory_is_packed =
      (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
       b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
      fmt == AGX_FORMAT_I8 && nir_src_num_components(instr->src[0]) > 1;
}

/*
 * Hardware bindless texture sources are specified as a 64-bit uniform base
 * address summed with a 32-bit register index. We model in NIR with the
 * bindless_image_agx intrinsic.
 */
static agx_index
agx_translate_bindless_handle(agx_builder *b, nir_src *handle, agx_index *base)
{
   nir_intrinsic_instr *intr = nir_src_as_intrinsic(*handle);
   assert(intr->intrinsic == nir_intrinsic_bindless_image_agx ||
          intr->intrinsic == nir_intrinsic_bindless_sampler_agx);

   *base = agx_uniform(nir_intrinsic_desc_set(intr), AGX_SIZE_64);
   return agx_src_index(&intr->src[0]);
}

/* Preambles write directly to uniform registers, so move from uniform to GPR */
static agx_instr *
agx_emit_load_preamble(agx_builder *b, agx_index dst,
                       nir_intrinsic_instr *instr)
{
   agx_index srcs[NIR_MAX_VEC_COMPONENTS];
   unsigned dim = instr->def.num_components;
   assert(dim <= ARRAY_SIZE(srcs) && "shouldn't see larger vectors");

   unsigned base = nir_intrinsic_base(instr);
   unsigned stride = agx_size_align_16(dst.size);

   for (unsigned i = 0; i < dim; ++i)
      srcs[i] = agx_uniform(base + i * stride, dst.size);

   return agx_emit_collect_to(b, dst, dim, srcs);
}

static agx_instr *
agx_emit_store_preamble(agx_builder *b, nir_intrinsic_instr *instr)
{
   nir_preamble_class cls = nir_intrinsic_preamble_class(instr);
   unsigned base = nir_intrinsic_base(instr);

   if (cls != nir_preamble_class_general) {
      agx_index heap, offset;
      offset = agx_translate_bindless_handle(b, &instr->src[0], &heap);

      /* base is 32-bit units for images but 16-bit for samplers, hence the
       * division difference to convert into texture/sampler state units.
       */
      if (cls == nir_preamble_class_image)
         return agx_tex_state_store(b, heap, offset, base / 2);
      else
         return agx_sampler_state_store(b, heap, offset, base);
   }

   agx_index vec = agx_src_index(&instr->src[0]);
   unsigned stride = agx_size_align_16(vec.size);
   unsigned nr = nir_src_num_components(instr->src[0]);

   /* An AGX3C uniform export is one 32-bit copy per word, so a 64-bit export
    * is two copies however it is spelled (agx3_pack_binary). Most 64-bit
    * fragment exports are texture and sampler handles built as
    * pack_64_2x32_split(word, 0): collecting the two words into an aligned
    * register pair first costs a move per word on top of the copies. Export
    * each word straight from its own register instead, into the same
    * uniform halves, in the same order and at the same program point. */
   nir_alu_instr *pack = nir_def_as_alu_or_null(instr->src[0].ssa);
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C && nr == 1 &&
       vec.size == AGX_SIZE_64 && pack &&
       pack->op == nir_op_pack_64_2x32_split) {
      for (unsigned w = 0; w < 2; ++w) {
         agx_index word = agx_extract_nir_src(b, pack->src[w].src,
                                              pack->src[w].swizzle[0]);
         agx_uniform_store(b, word, agx_immediate(base + 2 * w), 1);
      }

      return NULL;
   }

   for (unsigned i = 0; i < nr; i += (4 / stride)) {
      agx_index data[4] = {0};
      unsigned count = MIN2(4 / stride, nr - i);

      for (unsigned c = 0; c < count; ++c) {
         data[c] = agx_extract_nir_src(b, instr->src[0], i + c);
      }

      agx_uniform_store(b, agx_emit_collect(b, count, data),
                        agx_immediate(base + i * stride), BITFIELD_MASK(count));
   }

   return NULL;
}

static enum agx_dim
agx_tex_dim(enum glsl_sampler_dim dim, bool array)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D:
      return array ? AGX_DIM_1D_ARRAY : AGX_DIM_1D;

   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_EXTERNAL:
      return array ? AGX_DIM_2D_ARRAY : AGX_DIM_2D;

   case GLSL_SAMPLER_DIM_MS:
      return array ? AGX_DIM_2D_MS_ARRAY : AGX_DIM_2D_MS;

   case GLSL_SAMPLER_DIM_3D:
      assert(!array && "3D arrays unsupported");
      return AGX_DIM_3D;

   case GLSL_SAMPLER_DIM_CUBE:
      return array ? AGX_DIM_CUBE_ARRAY : AGX_DIM_CUBE;

   case GLSL_SAMPLER_DIM_BUF:
      UNREACHABLE("Buffer textures should have been lowered");

   default:
      UNREACHABLE("Invalid sampler dim\n");
   }
}

static agx_instr *
agx_emit_block_image_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index offset = agx_src_index(&instr->src[1]);
   agx_index coords = agx_src_index(&instr->src[2]);
   enum agx_format format = agx_format_for_pipe(nir_intrinsic_format(instr));

   bool ms = nir_intrinsic_image_dim(instr) == GLSL_SAMPLER_DIM_MS;
   bool array = nir_intrinsic_image_array(instr);
   enum agx_dim dim = agx_tex_dim(nir_intrinsic_image_dim(instr), array);
   bool explicit = nir_intrinsic_explicit_coord(instr);

   /* Modified coordinate descriptor */
   if (!explicit) {
      if (array) {
         agx_index layer = coords;
         coords = agx_temp(b->shader, AGX_SIZE_32);
         agx_emit_collect_to(b, coords, 2,
                             (agx_index[]){
                                ms ? agx_mov_imm(b, 16, 0) : layer,
                                ms ? layer : agx_undef(AGX_SIZE_16),
                             });
      } else {
         coords = agx_null();
      }
   }

   agx_index base, index;
   if (instr->intrinsic == nir_intrinsic_bindless_image_store_block_agx) {
      index = agx_translate_bindless_handle(b, &instr->src[0], &base);

      assert(base.size == AGX_SIZE_64);
      assert(index.size == AGX_SIZE_32);
   } else {
      base = agx_zero();
      index = agx_src_index(&instr->src[0]);

      assert(index.size == AGX_SIZE_16);
   }

   // XXX: how does this possibly work
   if (format == AGX_FORMAT_F16)
      format = AGX_FORMAT_I16;
   else if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
            format == AGX_FORMAT_SRGBA8)
      format = AGX_FORMAT_U8NORM;

   return agx_block_image_store(b, base, index, offset, coords, format, dim,
                                explicit);
}

static agx_instr *
agx_get_sr_native_width_to(agx_builder *b, agx_index dst, enum agx_sr sr)
{
   bool narrow_agx3 = false;

   switch (sr) {
   case AGX_SR_THREADS_PER_THREADGROUP_X:
   case AGX_SR_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_THREADS_PER_THREADGROUP_Z:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_X:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Z:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_X:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Y:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Z:
   case AGX_SR_SCS_THREAD_INDEX:
   case AGX_SR_THREAD_INDEX_IN_THREADGROUP:
   case AGX_SR_THREAD_INDEX_IN_SUBGROUP:
   case AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP:
   case AGX_SR_MESH_PAYLOAD_HI:
   case AGX_SR_MESH_PRIMITIVE_STORAGE_HI:
   case AGX_SR_BACKGROUND_LAYER_ID:
      narrow_agx3 = true;
      break;
   default:
      break;
   }

   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 && narrow_agx3 &&
       dst.size == AGX_SIZE_32) {
      /* Apple's AGX3 get_sr forms expose these builtins as witnessed 16-bit
       * values. NIR presents the Vulkan/D3D builtins as uint32_t, so read the
       * native half register and explicitly zero-extend it. This also makes
       * the async dependency visible to the scoreboard allocator. */
      agx_index native = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_sr_to(b, native, sr);
      return agx_mov_to(b, dst, native);
   }

   return agx_get_sr_to(b, dst, sr);
}

/* Apple reads the active-thread index/count of the SIMD-group (SR 34/36)
 * into half registers (witness-sr59); NIR presents them as uint32_t. */
static agx_instr *
agx_get_sr_coverage_native_width_to(agx_builder *b, agx_index dst,
                                    enum agx_sr sr)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       sr == AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP) {
      return agx_simd_reduce_to(b, dst,
                               agx_mov_imm(b, 32, 1), AGX_SIMD_OP_IADD);
   }
   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       dst.size == AGX_SIZE_32 &&
       (sr == AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP ||
        sr == AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP)) {
      agx_index native = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_sr_coverage_to(b, native, sr);
      return agx_mov_to(b, dst, native);
   }

   return agx_get_sr_coverage_to(b, dst, sr);
}

static agx_instr *
agx_load_compute_dimension(agx_builder *b, agx_index dst,
                           nir_intrinsic_instr *instr, enum agx_sr base)
{
   unsigned dim = instr->def.num_components;
   unsigned size = instr->def.bit_size;
   assert(size == 16 || size == 32);

   agx_index srcs[3];
   for (unsigned i = 0; i < ARRAY_SIZE(srcs); ++i) {
      srcs[i] = agx_temp(b->shader, agx_size_for_bits(size));
      agx_get_sr_native_width_to(b, srcs[i], base + i);
   }

   return agx_emit_collect_to(b, dst, dim, srcs);
}

static enum agx_atomic_opc
translate_atomic_opcode(nir_atomic_op op)
{
   /* clang-format off */
   switch (op) {
   case nir_atomic_op_iadd:    return AGX_ATOMIC_OPC_ADD;
   case nir_atomic_op_imin:    return AGX_ATOMIC_OPC_IMIN;
   case nir_atomic_op_umin:    return AGX_ATOMIC_OPC_UMIN;
   case nir_atomic_op_imax:    return AGX_ATOMIC_OPC_IMAX;
   case nir_atomic_op_umax:    return AGX_ATOMIC_OPC_UMAX;
   case nir_atomic_op_iand:    return AGX_ATOMIC_OPC_AND;
   case nir_atomic_op_ior:     return AGX_ATOMIC_OPC_OR;
   case nir_atomic_op_ixor:    return AGX_ATOMIC_OPC_XOR;
   case nir_atomic_op_xchg:    return AGX_ATOMIC_OPC_XCHG;
   case nir_atomic_op_cmpxchg: return AGX_ATOMIC_OPC_CMPXCHG;
   case nir_atomic_op_fadd:    return AGX_ATOMIC_OPC_FADD;
   /* Float atomics: the caller refuses the shader. */
   default: return (enum agx_atomic_opc)~0;
   }
   /* clang-format on */
}

/*
 * The "base" of a local load/store/atomic can be zero but no other immediates.
 * This would be a little silly to handle when inlining immediates, so we
 * instead exclude these ops from immediate inlining and just handle 0 specially
 * when translating.
 */
static agx_index
agx_local_base(nir_src src)
{
   if (nir_src_is_const(src) && nir_src_as_uint(src) == 0)
      return agx_zero();
   else
      return agx_src_index(&src);
}

static agx_index
agx_local_atomic_base(agx_builder *b, nir_src src)
{
   if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
       nir_src_is_const(src)) {
      const uint64_t address = nir_src_as_uint(src);

      /* G17P has a witnessed 16-bit immediate byte-address field for
       * threadgroup atomics, including returned exchanges at nonzero
       * addresses. Keep aligned constants immediate instead of needlessly
       * materializing the ungrounded returned register-address form. */
      if (address <= UINT16_MAX && (address & 3) == 0)
         return agx_immediate(address);
   }

   return agx_local_base(src);
}

static void
agx_emit_atomic(agx_builder *b, agx_index dst, nir_intrinsic_instr *instr,
                bool local)
{
   enum agx_atomic_opc op =
      translate_atomic_opcode(nir_intrinsic_atomic_op(instr));
   if (op == (enum agx_atomic_opc)~0) {
      fprintf(stderr, "Unhandled atomic op %u\n", nir_intrinsic_atomic_op(instr));
      b->shader->unsupported_instruction = true;
      return;
   }
   agx_index base =
      local ? agx_local_atomic_base(b, instr->src[0])
            : agx_src_index(&instr->src[0]);
   agx_index value = agx_src_index(&instr->src[local ? 1 : 2]);
   agx_index index = local ? agx_zero() : agx_src_index(&instr->src[1]);

   /* cmpxchg (only) takes 2 sources, passed in consecutive registers */
   if (op == AGX_ATOMIC_OPC_CMPXCHG) {
      agx_index value2 = agx_src_index(&instr->src[local ? 2 : 3]);
      value = agx_vec2(b, value2, value);
   }

   if (local) {
      assert(base.size == AGX_SIZE_16);
      agx_local_atomic_to(b, dst, value, base, index, op);
   } else {
      assert(base.size == AGX_SIZE_64);
      agx_atomic_to(b, dst, value, base, index, op);
   }
}

/* AGX3b device atomics return through their value register.  Legalize the
 * SSA three-address form before RA so the allocator can enforce that physical
 * two-address ABI without clobbering the independently evaluated index. */
bool
agx_lower_agx3_device_atomic_sources(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_ATOMIC || I->atomic_opc == AGX_ATOMIC_OPC_CMPXCHG ||
          I->nr_dests != 1 || I->nr_srcs != 3 || agx_is_null(I->dest[0]) ||
          I->dest[0].type != AGX_INDEX_NORMAL ||
          I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
          I->src[0].type != AGX_INDEX_NORMAL ||
          I->src[0].size != AGX_SIZE_32 || agx_channels(I->src[0]) != 1)
         continue;

      /* Native G17P atomicdev has one GPR field (field16) for both the input
       * value and returned old value.  Copy the SSA value to a last-use
       * temporary; RA then assigns the atomic destination to that temporary's
       * physical register.  The address index remains an independent live
       * operand and can never be overwritten by this legalization. */
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index tied = agx_temp(ctx, AGX_SIZE_32);
      agx_mov_to(&b, tied, I->src[0]);
      I->src[0] = tied;
      progress = true;
   }

   return progress;
}

static bool
agx_lower_agx3_matrix_sources(agx_context *ctx)
{
   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_MATMAD16_AGX3)
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index tied = agx_vec_temp(ctx, AGX_SIZE_32, 8);
      agx_mov_to(&b, tied, I->src[2]);
      I->src[2] = tied;
      progress = true;
   }
   return progress;
}

static enum agx_format
format_for_bitsize(unsigned bitsize)
{
   switch (bitsize) {
   case 8:
      return AGX_FORMAT_I8;
   case 16:
      return AGX_FORMAT_I16;
   case 32:
      return AGX_FORMAT_I32;
   default:
      UNREACHABLE("should've been lowered");
   }
}

/* AGX3 ldtg/sttg carry an immediate byte offset next to the element index.
 * agx_nir_opt_shared_byte_offsets() folds a constant addend of the byte
 * address into the intrinsic's base; it rides in the base operand here
 * (agx3_translate_local() moves it into the descriptor's byte_offset).
 * AGX2 keeps its byte-addressed base register and never sees a nonzero base.
 */
static agx_index
agx_local_immediate_base(agx_builder *b, nir_intrinsic_instr *instr)
{
   const unsigned base = nir_intrinsic_base(instr);
   assert(base == 0 ||
          b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2);
   if (base == 0)
      return agx_zero();

   agx_index imm = agx_immediate(base);
   imm.size = AGX_SIZE_16;
   return imm;
}

static void
agx_emit_local_load(agx_builder *b, agx_index dst, nir_intrinsic_instr *instr)
{
   agx_index address = agx_local_base(instr->src[0]);
   const bool agx3 = b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
   /* Shared offsets are relative to the workgroup allocation. */
   agx_index base = agx3 ? agx_local_immediate_base(b, instr) : address;
   agx_index index = agx3 ? address : agx_zero();
   assert(address.size == AGX_SIZE_16);

   enum agx_format format = format_for_bitsize(instr->def.bit_size);
   unsigned nr = instr->def.num_components;
   unsigned mask = BITFIELD_MASK(nr);

   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       format == AGX_FORMAT_I8 && nr > 1) {
      /* LDTG returns packed bytes, not one byte in each half register.
       * Keep the vector memory transaction and unpack its physical payload. */
      const unsigned nr_halves = DIV_ROUND_UP(nr, 2);
      agx_index packed = agx_vec_temp(b->shader, AGX_SIZE_16, nr_halves);
      agx_local_load_to(b, packed, base, index, format, mask)->memory_is_packed =
         true;
      agx_index halves[2] = {agx_null()};
      agx_emit_split(b, halves, packed, nr_halves);
      agx_unpack_integer_8_agx3(b, dst, halves, nr);
      agx_emit_cached_split(b, dst, nr);
      return;
   }

   agx_local_load_to(b, dst, base, index, format, mask);
   agx_emit_cached_split(b, dst, nr);
}

static void
agx_emit_local_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index value = agx_src_index(&instr->src[0]);
   agx_index address = agx_local_base(instr->src[1]);
   const bool agx3 = b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
   agx_index base = agx3 ? agx_local_immediate_base(b, instr) : address;
   agx_index index = agx3 ? address : agx_zero();
   assert(address.size == AGX_SIZE_16);

   enum agx_format format = format_for_bitsize(nir_src_bit_size(instr->src[0]));
   unsigned mask = BITFIELD_MASK(
      nir_src_num_components(instr->src[0])); /* XXX: there's a write mask */

   const bool packed_bytes =
      b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
      format == AGX_FORMAT_I8 && nir_src_num_components(instr->src[0]) > 1;
   if (packed_bytes) {
      const unsigned nr = nir_src_num_components(instr->src[0]);
      agx_index channels[4] = {agx_null()};
      for (unsigned i = 0; i < nr; ++i)
         channels[i] = agx_extract_nir_src(b, instr->src[0], i);
      agx_index packed[2] = {agx_null()};
      const unsigned nr_packed =
         agx_pack_integer_8_agx3(b, channels, nr, format, packed);
      assert(nr_packed == DIV_ROUND_UP(nr, 2));
      value = agx_emit_collect(b, nr_packed, packed);
   }
   agx_local_store(b, value, base, index, format, mask)->memory_is_packed =
      packed_bytes;
}

/* Scalarize static 32-bit stack vectors after RA so both NIR scratch and
 * allocator spills use the native scalar TPS copy/rld protocol. No temporary
 * GPR is required: component registers and frame offsets are already fixed. */

static void
agx_lower_agx3_stack_vectors(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      const bool store = I->op == AGX_OPCODE_STACK_STORE;
      if ((!store && I->op != AGX_OPCODE_STACK_LOAD) ||
          (I->format != AGX_FORMAT_I32 && I->format != AGX_FORMAT_I16) ||
          I->mask == 1)
         continue;
      /* Spilled 16-bit vectors take the same route one half at a time
       * (subgroups.ballot_broadcast.*.subgroupbroadcast_{f,i,u}16vec*). */
      const bool halves = I->format == AGX_FORMAT_I16;
      agx_index data = store ? I->src[0] : I->dest[0];
      agx_index offset = I->src[store ? 1 : 0];
      if (data.type != AGX_INDEX_REGISTER || data.memory ||
          (halves ? data.size != AGX_SIZE_16
                  : (data.size != AGX_SIZE_32 && data.size != AGX_SIZE_64)) ||
          offset.type != AGX_INDEX_IMMEDIATE)
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      unsigned packed = 0;
      u_foreach_bit(c, I->mask) {
         agx_index component = data;
         component.value += (halves ? 1 : 2) * packed++;
         component.channels_m1 = 0;
         component.size = halves ? AGX_SIZE_16 : AGX_SIZE_32;
         agx_index address = agx_immediate(offset.value + (halves ? 2 : 4) * c);
         if (store)
            agx_stack_store(&b, component, address, I->format, 1);
         else
            agx_stack_load_to(&b, component, address, I->format, 1);
      }
      agx_remove_instruction(I);
   }
}

static void
agx_emit_load_scratch(agx_builder *b, agx_index dst, nir_intrinsic_instr *instr)
{
   agx_index offset = agx_src_index(&instr->src[0]);
   enum agx_format format = format_for_bitsize(instr->def.bit_size);
   unsigned nr = instr->def.num_components;
   unsigned mask = BITFIELD_MASK(nr);

   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       format == AGX_FORMAT_I8 && nr > 1) {
      const unsigned n = DIV_ROUND_UP(nr, 2);
      agx_index packed = agx_vec_temp(b->shader, AGX_SIZE_16, n);
      agx_stack_load_to(b, packed, offset, format, mask)->memory_is_packed = true;
      agx_index halves[2] = {agx_null()};
      agx_emit_split(b, halves, packed, n);
      agx_unpack_integer_8_agx3(b, dst, halves, nr);
   } else {
      agx_stack_load_to(b, dst, offset, format, mask);
   }
   agx_emit_cached_split(b, dst, nr);
   b->shader->any_scratch = true;
}

static void
agx_emit_store_scratch(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index value = agx_recollect_vector(b, instr->src[0]);
   agx_index offset = agx_src_index(&instr->src[1]);
   enum agx_format format = format_for_bitsize(nir_src_bit_size(instr->src[0]));
   unsigned mask = BITFIELD_MASK(nir_src_num_components(instr->src[0]));

   const unsigned nr = nir_src_num_components(instr->src[0]);
   const bool packed_bytes =
      b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
      format == AGX_FORMAT_I8 && nr > 1;
   if (packed_bytes) {
      agx_index channels[4] = {agx_null()}, packed[2] = {agx_null()};
      for (unsigned i = 0; i < nr; ++i)
         channels[i] = agx_extract_nir_src(b, instr->src[0], i);
      const unsigned n = agx_pack_integer_8_agx3(b, channels, nr, format, packed);
      value = agx_emit_collect(b, n, packed);
   }
   agx_stack_store(b, value, offset, format, mask)->memory_is_packed = packed_bytes;
   b->shader->any_scratch = true;
}

static unsigned
agx_expand_tex_to(agx_builder *b, nir_def *def, agx_index src, bool masked)
{
   unsigned nr_channels = def->num_components;
   nir_component_mask_t mask = nir_def_components_read(def);

   if (!masked)
      mask = (nir_component_mask_t)BITFIELD_MASK(nr_channels);

   agx_index packed_channels[8] = {agx_null()};
   agx_index unpacked_channels[8] = {agx_null()};

   /* Hardware writes the masked components contiguously, expand out for NIR */
   agx_emit_split(b, packed_channels, src,
                  ALIGN_POT(nr_channels, 4) /* XXX: why not nr_channels */);

   for (unsigned i = 0; i < nr_channels; ++i) {
      unpacked_channels[i] =
         (mask & BITFIELD_BIT(i))
            ? packed_channels[util_bitcount(mask & BITFIELD_MASK(i))]
            : agx_undef(src.size);
   }

   agx_emit_collect_to(b, agx_def_index(def), nr_channels, unpacked_channels);
   return mask;
}

/* G18G texel fetches and image reads are witnessed with one channel or all
 * four (the result-mask gate in agx3_emit_texture_fetch_g17g). They clobber
 * four registers whatever the mask (agx_validate.c), so a partial read writes
 * all four instead and the unread channels are simply dead. */
static bool
agx_fetch_masked(agx_builder *b, nir_def *def, bool masked)
{
   return masked &&
          !(agx_is_g18g(&b->shader->key->dev) &&
            nir_def_components_read(def) != 1);
}

static agx_instr *
agx_emit_image_load(agx_builder *b, agx_index dst, nir_intrinsic_instr *intr)
{
   agx_index ms_index = agx_src_index(&intr->src[2]);
   agx_index lod = agx_src_index(&intr->src[3]);
   enum agx_lod_mode lod_mode = AGX_LOD_MODE_LOD_MIN;
   bool sparse = intr->intrinsic == nir_intrinsic_image_sparse_load ||
                 intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;

   agx_index bindless = agx_immediate(0), texture;
   if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {

      texture = agx_translate_bindless_handle(b, &intr->src[0], &bindless);
   } else if (nir_src_is_const(intr->src[0]) &&
              nir_src_as_uint(intr->src[0]) < 0x100) {
      texture = agx_immediate(nir_src_as_uint(intr->src[0]));
   } else {
      texture = agx_src_index(&intr->src[0]);
   }

   assert(nir_src_num_components(intr->src[1]) == 4);
   agx_index coord[4] = {
      agx_extract_nir_src(b, intr->src[1], 0),
      agx_extract_nir_src(b, intr->src[1], 1),
      agx_extract_nir_src(b, intr->src[1], 2),
      agx_extract_nir_src(b, intr->src[1], 3),
   };

   /* Get the image dimension. Cubes are lowered to 2D, since they are logically
    * equivalent for imageLoad, but out-of-bounds behaviour for cubes on G13
    * is wrong according to Piglit's arb_shader_image_load_store-invalid.
    *
    * This requires a matching transform in the driver.
    */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool is_array = nir_intrinsic_image_array(intr);

   if (dim == GLSL_SAMPLER_DIM_CUBE) {
      dim = GLSL_SAMPLER_DIM_2D;
      is_array = true;
   }

   bool is_ms = dim == GLSL_SAMPLER_DIM_MS;
   unsigned coord_comps = glsl_get_sampler_dim_coordinate_components(dim);
   if (is_array && is_ms) {
      agx_index layer = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, layer, coord[coord_comps], 0);

      assert(ms_index.size == AGX_SIZE_16);
      agx_index tmp = agx_temp(b->shader, AGX_SIZE_32);
      agx_emit_collect_to(b, tmp, 2, (agx_index[]){ms_index, layer});
      coord[coord_comps++] = tmp;
   } else if (is_ms) {
      agx_index tmp = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, tmp, ms_index);
      coord[coord_comps++] = tmp;
   } else if (is_array) {
      coord_comps++;
   }

   /* Multisampled images do not support mipmapping */
   if (is_ms) {
      lod_mode = AGX_LOD_MODE_AUTO_LOD;
      lod = agx_zero();
   } else if ((b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
               b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
              (dim == GLSL_SAMPLER_DIM_2D ||
               (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
                dim == GLSL_SAMPLER_DIM_3D)) &&
              !is_array && !sparse &&
              nir_src_is_const(intr->src[3]) &&
              nir_src_as_uint(intr->src[3]) == 0) {
      /* Match texelFetch's constant-level handling. Image reads also use
       * native SMP, whose immediate level-zero form needs no PACK_SMPIR
       * staging. This applies to heap-indexed storage images and all formats;
       * neither the resource operand nor the component type changes the LOD.
       *
       * G18G too: native run112 kick 19's texture2d<uint, access::read>
       * kernel reads with the 14-byte immediate-level smp (060145c0...,
       * ep1), the form Mesa's own G18G texelFetch already takes, not the
       * PACK_SMPIR + 24-byte register-level smp a run-time zero level
       * costs. */
      lod_mode = AGX_LOD_MODE_AUTO_LOD;
      lod = agx_zero();
   }

   /* G17P's dynamic integer-level PACK_SMPIR reads a 32-bit GPR. Apple's
    * native ushort-level image kernel widens the loaded ushort before the
    * otherwise byte-identical staging/fetch sequence. Robust bindless image
    * lowering can synthesize the same semantic uint level in a 16-bit
    * register, so make that required widening explicit before RA. */
   if ((b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
       lod_mode == AGX_LOD_MODE_LOD_MIN && lod.size == AGX_SIZE_16 &&
       lod.type != AGX_INDEX_IMMEDIATE) {
      agx_index lod32 = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, lod32, lod);
      lod = lod32;
   }

   /* G17P indexed-image lowering puts the native heap index in the 64-bit
    * resource operand. Its SMP encoding does not consume the legacy uniform
    * base. Drop that bookkeeping operand before CProg input import, so it
    * does not become an unused register operand or an extra TPS load.
    */
   if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
       texture.size == AGX_SIZE_64)
      bindless = agx_zero();

   agx_index coords = agx_emit_collect(b, coord_comps, coord);
   agx_index tmp = agx_vec_temp(b->shader, dst.size, sparse ? 8 : 4);

   agx_instr *I = agx_image_load_to(b, tmp, coords, lod, bindless, texture,
                                    agx_immediate(0), agx_null(),
                                    agx_tex_dim(dim, is_array), lod_mode, 0,
                                    false, sparse, nir_is_coherent(intr));
   /* ACCESS_CAN_REORDER (nir_opt_access: the image is NonWritable, so
    * nothing in this shader writes it) and ACCESS_RESTRICT (vtn: an image
    * not decorated Aliased, which SPIR-V's Simple/GLSL/Vulkan memory models
    * let the consumer assume does not alias another memory object
    * declaration) together mean no store of this shader can write what the
    * load reads, so the M5 copy schedule may move it across them. M5 encodes
    * the read as a native SMP, which Apple's compiler issues ahead of earlier
    * image writes (native725 vs_copy). */
   const enum gl_access_qualifier access = nir_intrinsic_access(intr);
   I->memory_can_reorder =
      b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
      (access & ACCESS_CAN_REORDER) && (access & ACCESS_RESTRICT) &&
      !nir_is_coherent(intr);
   I->mask = agx_expand_tex_to(b, &intr->def, tmp,
                               agx_fetch_masked(b, &intr->def, !sparse));

   b->shader->out->uses_txf = true;
   b->shader->out->sampler_state_count =
      MAX2(b->shader->out->sampler_state_count, 1);
   return NULL;
}

static agx_instr *
agx_emit_export(agx_builder *b, unsigned base, nir_src src)
{
   agx_builder b_ = *b;
   agx_cursor after_cursor = agx_after_block(agx_exit_block(b->shader));
   b_.cursor = after_cursor;

   for (unsigned c = 0; c < nir_src_num_components(src); ++c) {
      agx_index chan = agx_extract_nir_src(b, src, c);
      unsigned stride = agx_size_align_16(chan.size);

      agx_export(&b_, chan, base + (c * stride));
   }

   if (agx_cursors_equal(b->cursor, after_cursor)) {
      b->cursor = agx_after_block_logical(b->cursor.block);
   }

   return NULL;
}

static agx_instr *
agx_load_exported_to(agx_builder *b, agx_index dst, unsigned base, unsigned nr)
{
   agx_index chans[4] = {0};
   unsigned stride = agx_size_align_16(dst.size);

   for (unsigned c = 0; c < nr; ++c) {
      chans[c] = agx_cached_preload(b->shader, base + c * stride, dst.size);
   }

   return agx_emit_collect_to(b, dst, nr, chans);
}

static agx_instr *
agx_emit_image_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   /* See remarks in agx_emit_image_load */
   enum glsl_sampler_dim glsl_dim = nir_intrinsic_image_dim(instr);
   bool is_array = nir_intrinsic_image_array(instr);

   if (glsl_dim == GLSL_SAMPLER_DIM_CUBE) {
      glsl_dim = GLSL_SAMPLER_DIM_2D;
      is_array = true;
   }

   enum agx_dim dim = agx_tex_dim(glsl_dim, is_array);
   assert(glsl_dim != GLSL_SAMPLER_DIM_MS && "needs to be lowered");

   agx_index base, index;
   if (instr->intrinsic == nir_intrinsic_bindless_image_store) {
      index = agx_translate_bindless_handle(b, &instr->src[0], &base);

      assert(base.size == AGX_SIZE_64);
      assert(index.size == AGX_SIZE_32);
   } else {
      base = agx_zero();
      index = agx_src_index(&instr->src[0]);

      assert(index.size == AGX_SIZE_16);
   }

   agx_index coords4 = agx_src_index(&instr->src[1]);
   agx_index lod = agx_src_index(&instr->src[4]);
   assert(lod.size == AGX_SIZE_16);

   int coord_components = glsl_get_sampler_dim_coordinate_components(glsl_dim);
   if (is_array)
      coord_components++;

   agx_index coord_comps[4] = {};
   for (unsigned i = 0; i < coord_components; ++i)
      coord_comps[i] = agx_emit_extract(b, coords4, i);

   agx_index coords = agx_emit_collect(b, coord_components, coord_comps);
   agx_index data = agx_src_index(&instr->src[3]);

   /* If the image format has less than 4 components, nir_opt_shrink_stores can
    * shrink the store. But the IR still expects 4 components: pad with undef.
    */
   if (nir_src_num_components(instr->src[3]) < 4) {
      agx_index chan[4] = {agx_null()};

      for (unsigned i = 0; i < 4; ++i) {
         if (i < nir_src_num_components(instr->src[3]))
            chan[i] = agx_extract_nir_src(b, instr->src[3], i);
         else
            chan[i] = agx_undef(data.size);
      }

      data = agx_emit_collect(b, 4, chan);
   }

   /* Image stores act like tilebuffer stores when used for tib spilling */
   b->shader->out->tag_write_disable = false;

   agx_instr *write = agx_image_write(b, data, coords, lod, base, index, dim,
                                      nir_is_coherent(instr));
   /* The selector below is native G17P's witnessed imgwci f70 rule. The
    * AGX3C packer encodes its float/UNORM and I32 forms for 1D/2D stores
    * only; 3D, array and cube stores keep the integer format it encodes for
    * every dimension. Selecting UNORM for those made every float 3D, array
    * and cube store unencodable on M5 (image.format_reinterpret.3d,
    * geometry.layered secondary_cmd_buffer, image.store.without_format:
    * Pass on the pre-widening build, packer refusals after it), while 2D
    * stores pass with it (glsl.builtin_var.fragcoord_msaa*). */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
       (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
        (dim == AGX_DIM_1D || dim == AGX_DIM_2D))) {
      /* imgwci.f70 follows the TYPE OF THE DATA the shader stores, not the
       * destination surface's format. Metal never puts a pixel format in a
       * shader -- a texture2d<float, access::write> is written the same way
       * whichever surface is bound -- so f70 cannot be encoding one. Apple's
       * own applegpu_g17p output for four kernels that differ only in the
       * texture's data type, each `t.write(v, gid)`:
       *
       *   texture2d<float> d71054000040000000040000883a0900   f70:1 f71:2
       *   texture2d<uint>  d71054000040000000040000881a0900   f70:0 f71:2
       *   texture2d<int>   d71054000040000000040000881a0900   f70:0 f71:2
       *   texture2d<half>  d71054000040000000040000882a0900   f70:1 f71:1
       *
       * int and uint are byte-identical, so f70 separates float from integer
       * data; f71 is the data width, which the packer pins to 32-bit.
       *
       * Selecting on nir_intrinsic_format() reproduced that only for the
       * three formats spelled out here and failed every other one closed --
       * including a format-less image, which is what Vulkan's
       * shaderStorageImageWriteWithoutFormat gives and what DXVK's clear and
       * copy kernels all declare. Select on the stored type instead: it
       * gives the same f70 for those three formats and is the rule Apple's
       * compiler is following. */
      nir_alu_type stored =
         nir_alu_type_get_base_type(nir_intrinsic_src_type(instr));
      if (stored == nir_type_invalid) {
         /* Not every producer of an image store fills in src_type. Fall back
          * to the declared format, which says the same thing whenever it is
          * there at all. */
         switch (nir_intrinsic_format(instr)) {
         case PIPE_FORMAT_R8G8B8A8_UINT:
         case PIPE_FORMAT_R32_UINT:
            stored = nir_type_uint;
            break;
         case PIPE_FORMAT_R8G8B8A8_SINT:
         case PIPE_FORMAT_R32_SINT:
            stored = nir_type_int;
            break;
         case PIPE_FORMAT_R32_FLOAT:
         case PIPE_FORMAT_R8G8B8A8_UNORM:
         case PIPE_FORMAT_R10G10B10A2_UNORM:
            stored = nir_type_float;
            break;
         default:
            /* Any other declared format names its data class the same way
             * (the G18G branch below already does this): the PBE converts
             * the four 32-bit lanes, f70 only says float or integer. */
            if (nir_intrinsic_format(instr) != PIPE_FORMAT_NONE)
               stored = util_format_is_pure_integer(nir_intrinsic_format(instr))
                           ? nir_type_int : nir_type_float;
            break;
         }
      }

      switch (stored) {
      case nir_type_float:
         write->format = AGX_FORMAT_U8NORM;
         break;
      case nir_type_int:
      case nir_type_uint:
         write->format = AGX_FORMAT_I8;
         break;
      default:
         write->format = AGX_NUM_FORMATS;
         break;
      }
   } else if (b->shader->key->dev.emit_agx3d) {
      /* G18G's writes (agx3_emit_storage_image_write_g18g). An R32 integer
       * image keeps native run112 kick 19's one-component form, which stores
       * one 32-bit integer component and zeroes the rest: the whole texel
       * only for a one-channel 32-bit integer format.
       *
       * Every other write stores four components, and as on G17P above it
       * is the TYPE OF THE DATA that selects imgwci f70, never the image's
       * format. metal-nt -arch applegpu_g18g writing float4, uint4 and int4
       * to one texture2d<T, access::write> (m6-image-write iw_cs_rgba32f,
       * m6-image-write2 iw2_{u4,i4}_t0, and the same through texture(1),
       * (2), (5) and (14)):
       *
       *   float4  160940c017002100008010000020  f70 1
       *   uint4   160940c017002300008000000020  f70 0
       *   int4    160940c017002300008000000020  f70 0, byte-identical
       *
       * so a format-less writeonly image, which GL allows, is written the
       * same way as a formatted one. The markers are the G17P branch's:
       * AGX_FORMAT_U8NORM for float data, AGX_FORMAT_I8 for integer data.
       * A store that names no type falls back to its declared format's
       * class; with neither, it stays unnamed and the packer refuses it. */
      const enum pipe_format format = nir_intrinsic_format(instr);
      nir_alu_type stored =
         nir_alu_type_get_base_type(nir_intrinsic_src_type(instr));
      if (stored == nir_type_invalid && format != PIPE_FORMAT_NONE)
         stored = util_format_is_pure_integer(format) ? nir_type_int
                                                      : nir_type_float;
      if ((format == PIPE_FORMAT_R32_UINT || format == PIPE_FORMAT_R32_SINT) &&
          (stored == nir_type_uint || stored == nir_type_int))
         write->format = AGX_FORMAT_I32;
      else if (stored == nir_type_float)
         write->format = AGX_FORMAT_U8NORM;
      else if (stored == nir_type_uint || stored == nir_type_int)
         write->format = AGX_FORMAT_I8;
      else
         write->format = AGX_NUM_FORMATS;
   }
   return write;
}

static enum agx_simd_op
translate_simd_op(nir_op op)
{
#define CASE(agx_, nir_)                                                       \
   case nir_op_##nir_:                                                         \
      return AGX_SIMD_OP_##agx_;

   switch (op) {
      CASE(AND, iand)
      CASE(FADD, fadd)
      CASE(OR, ior)
      CASE(FMUL, fmul)
      CASE(XOR, ixor)
      CASE(FMIN, fmin)
      CASE(FMAX, fmax)
      CASE(IADD, iadd)
      CASE(SMIN, imin)
      CASE(SMAX, imax)
      CASE(UMIN, umin)
      CASE(UMAX, umax)
   default:
      UNREACHABLE("unknown simd op");
   }
#undef CASE
}

static agx_index
agx_g18_sample_mask_word(agx_builder *b, agx_index affected, agx_index keep,
                         bool no_tests)
{
   /* Native G18 one-sample discard supplies MOVMSK with coverage in bits
    * 0 and 8, and the live sample in bit 24. Its FSELI16 selects 0 or the
    * static uniform halfword 0x0100 into the high half of the staging word.
    * These are byte fields, not three nibbles in the low halfword.
    */
   agx_index zero = agx_immediate(0);
   /* Native early kills pass only the affected byte (zero is a no-op).
    * A full update must also be zero when no sample is affected; main
    * shader and epilog must not run tests twice on surviving samples. */
   agx_index low = no_tests ? affected :
      agx_iadd(b, affected, agx_iadd(b, zero, affected, 4), 4);
   agx_index high = no_tests ? agx_mov_imm(b, 16, 0) :
      agx_iadd(b, zero, agx_iadd(b, zero, keep, 4), 4);
   agx_index word = agx_temp(b->shader, AGX_SIZE_32);
   agx_emit_collect_to(b, word, 2, (agx_index[]){low, high});
   return word;
}

/* A dynamic upper byte does not make the native target byte partial.
 * Alpha-to-coverage epilogs can leave TARGET as (0xff | dynamic16).
 * Prove the scalar byte without weakening the partial-update protocol. */
static bool
agx_g18_all_sample_targets(nir_src target)
{
   if (nir_src_is_const(target))
      return nir_src_as_uint(target) == 0xff;

   if (target.ssa->num_components != 1 || target.ssa->bit_size != 16)
      return false;

   nir_alu_instr *alu = nir_src_as_alu(target);
   if (!alu || alu->op != nir_op_ior)
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      nir_alu_src *src = &alu->src[i];
      if (src->src.ssa->num_components == 1 && src->swizzle[0] == 0 &&
          nir_src_is_const(src->src) &&
          (nir_src_as_uint(src->src) & 0xffu) == 0xffu)
         return true;
   }

   return false;
}

/* Native1121 psdiscard updates one sample per loop iteration. Admit the
 * unrolled equivalent only: a top-level update whose 16-bit TARGET is masked
 * to exactly one of the framebuffer's samples and whose LIVE is dynamic. */
static bool
agx_g18_single_sample_target(nir_intrinsic_instr *instr, unsigned samples)
{
   if (instr->instr.block->cf_node.parent->type != nir_cf_node_function ||
       instr->src[0].ssa->num_components != 1 ||
       instr->src[0].ssa->bit_size != 16 ||
       instr->src[1].ssa->num_components != 1 ||
       instr->src[1].ssa->bit_size != 16 || nir_src_is_const(instr->src[1]))
      return false;

   nir_alu_instr *mask = nir_src_as_alu(instr->src[0]);
   if (!mask || mask->op != nir_op_iand)
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      nir_alu_src *src = &mask->src[i];
      if (src->src.ssa->num_components == 1 && src->swizzle[0] == 0 &&
          nir_src_is_const(src->src)) {
         const uint64_t bits = nir_src_as_uint(src->src);
         return util_bitcount64(bits) == 1 &&
                bits < (UINT64_C(1) << samples);
      }
   }

   return false;
}

/* A load_scs_*_agx carrying AGX_NIR_SCS_TOUCH: access the address without a
 * destination. The intrinsic's value is never defined. */
static agx_instr *
agx_emit_scs_touch(agx_builder *b, nir_intrinsic_instr *instr, bool indexed)
{
   const unsigned flags = nir_intrinsic_flags(instr);
   const unsigned mask = BITFIELD_MASK(instr->def.num_components);
   const agx_index base = agx_src_index(&instr->src[0]);

   if (flags & AGX_NIR_SCS_TOUCH_FENCE_BEFORE)
      agx_rt_barrier_gen1(b, AGX3_RT_BARRIER_ENTRY);

   agx_instr *touch = indexed
      ? agx_scs_touch_register_indexed_agx3(
           b, base, agx_src_index(&instr->src[1]), nir_intrinsic_base(instr),
           mask, nir_intrinsic_component(instr))
      : agx_scs_touch_register_base_agx3(b, base, nir_intrinsic_base(instr),
                                         mask);

   if (flags & AGX_NIR_SCS_TOUCH_FENCE_AFTER)
      agx_rt_barrier_gen1(b, AGX3_RT_BARRIER_IN_LOOP);

   return touch;
}

static agx_instr *
agx_emit_intrinsic(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index dst = nir_intrinsic_infos[instr->intrinsic].has_dest
                      ? agx_def_index(&instr->def)
                      : agx_null();
   mesa_shader_stage stage = b->shader->stage;

   switch (instr->intrinsic) {
   case nir_intrinsic_cmat_muladd_agx: {
      unsigned type = nir_intrinsic_src_type(instr) == nir_type_uint16
         ? AGX3_MATRIX_INPUT_BF16 : nir_intrinsic_src_type(instr) == nir_type_float32
         ? AGX3_MATRIX_INPUT_F32 : AGX3_MATRIX_INPUT_F16;
      unsigned type_b = nir_intrinsic_src_type2(instr) == nir_type_uint16
         ? AGX3_MATRIX_INPUT_BF16 : nir_intrinsic_src_type2(instr) == nir_type_float32
         ? AGX3_MATRIX_INPUT_F32 : AGX3_MATRIX_INPUT_F16;
      type |= type_b << 4;
      /* The 16x16 GEMM form is M5 (AGX3C) only; the 8x8 matmad also exists on
       * the AGX3B encoding (G15/G16/G17P), so the 2-component path is allowed
       * there too. */
      const enum agx_isa_encoding mm_enc = b->shader->key->dev.encoding;
      assert(stage == MESA_SHADER_COMPUTE && !b->shader->is_preamble &&
             (mm_enc == AGX_ISA_ENCODING_AGX3C ||
              (mm_enc == AGX_ISA_ENCODING_AGX3B &&
               instr->def.num_components == 2)));
      if (instr->def.num_components == 8) {
         agx_matmad16_agx3_to(b, dst, agx_src_index(&instr->src[0]),
                            agx_src_index(&instr->src[1]),
                            agx_src_index(&instr->src[2]), type);
      } else {
         assert(instr->def.num_components == 2);
         agx_matmad8_agx3_to(b, dst, agx_src_index(&instr->src[0]),
                            agx_src_index(&instr->src[1]),
                            agx_src_index(&instr->src[2]), type);
      }
      agx_emit_cached_split(b, dst, instr->def.num_components);
      return NULL;
   }
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_at_sample:
   case nir_intrinsic_load_barycentric_at_offset:
      /* handled later via load_vary */
      return NULL;
   case nir_intrinsic_load_interpolated_input:
      assert(stage == MESA_SHADER_FRAGMENT);
      agx_emit_load_vary(b, dst, instr);
      return NULL;

   case nir_intrinsic_load_coefficients_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      agx_ldcf_to(b, dst, cf_for_intrinsic(b, instr), 1);
      agx_emit_cached_split(b, dst, 3);
      return NULL;

   case nir_intrinsic_load_agx:
   case nir_intrinsic_load_constant_agx:
      agx_emit_load(b, dst, instr);
      return NULL;

   case nir_intrinsic_store_uvs_agx:
      assert(stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_EVAL);
      return agx_st_vary(b, agx_src_index(&instr->src[1]),
                         agx_src_index(&instr->src[0]));

   case nir_intrinsic_store_agx:
      agx_emit_store(b, instr);
      return NULL;

   case nir_intrinsic_store_shared:
      agx_emit_local_store(b, instr);
      return NULL;

   case nir_intrinsic_load_shared:
      agx_emit_local_load(b, dst, instr);
      return NULL;

   case nir_intrinsic_global_atomic_agx:
   case nir_intrinsic_global_atomic_swap_agx:
      b->shader->out->uses_global_atomics = true;
      agx_emit_atomic(b, dst, instr, false);
      return NULL;

   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
      agx_emit_atomic(b, dst, instr, true);
      return NULL;

   case nir_intrinsic_store_zs_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      return agx_emit_store_zs(b, instr);

   case nir_intrinsic_zs_transaction_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      if (nir_intrinsic_flags(instr) == 0) {
         b->shader->out->writes_sample_mask = true;
         agx_wait_pixel_mask(b, 0x1);
         return NULL;
      }
      return agx_signal_pix(b, 0x1);

   case nir_intrinsic_store_local_pixel_agx:
      return agx_emit_local_store_pixel(b, instr);

   case nir_intrinsic_load_local_pixel_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      agx_emit_local_load_pixel(b, dst, instr);
      return NULL;

   case nir_intrinsic_load_pixel_coord:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
         return agx_emit_collect_to(
            b, dst, 2,
            (agx_index[2]){
               agx_get_sr(b, 16, AGX_SR_THREAD_POSITION_IN_GRID_X),
               agx_get_sr(b, 16, AGX_SR_THREAD_POSITION_IN_GRID_Y),
            });
      } else {
         agx_index xy[2];
         for (unsigned i = 0; i < 2; i++) {
            agx_index native = agx_temp(b->shader, AGX_SIZE_32);
            agx_get_sr_native_width_to(
               b, native, AGX_SR_THREAD_POSITION_IN_GRID_X + i);
            xy[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_subdivide_to(b, xy[i], native, 0);
            b->shader->agx3_pixel_coord_native[i] = native;
            b->shader->agx3_pixel_coord_half[i] = xy[i];
         }
         return agx_emit_collect_to(b, dst, 2, xy);
      }

   case nir_intrinsic_load_frag_coord_z: {
      agx_index cf = agx_get_cf(b->shader, VARYING_SLOT_POS, 2);

      /* AGX3 has no legacy ITER. G18G position.z/w use ordinary scalar
       * interpolation too, with the native f11 mode set. This is a direct
       * coefficient value, without a software perspective divide. */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
         return agx_interp_agx3_to(b, dst, cf, agx_zero(),
                                     AGX_INTERPOLATION_CENTER,
                                     agx_is_g18g(&b->shader->key->dev));

      return agx_iter_to(b, dst, cf, agx_zero(), 1, AGX_INTERPOLATION_CENTER);
   }

   case nir_intrinsic_load_frag_coord_w: {
      agx_index cf = agx_get_cf(b->shader, VARYING_SLOT_POS, 3);

      /* Native position.w uses ordinary interpolation (f54=0). The
       * projective RHW form is private to reciprocal interpolation and
       * cannot be reused as a public arithmetic value. */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
         return agx_interp_agx3_to(b, dst, cf, agx_zero(),
                                     AGX_INTERPOLATION_CENTER,
                                     agx_is_g18g(&b->shader->key->dev));

      return agx_iter_to(b, dst, cf, agx_zero(), 1, AGX_INTERPOLATION_CENTER);
   }

   case nir_intrinsic_sample_mask_agx: {
      assert(stage == MESA_SHADER_FRAGMENT);
      /* TARGET selects the samples whose coverage or Z/S tests may change.
       * An empty target has no effect, including after per-sample lowering.
       * Do not send this identity through the legacy SAMPLE_MASK encoder. */
      if (b->shader->key->dev.emit_agx3d &&
          (b->shader->key->fs.nr_samples == 2 ||
           b->shader->key->fs.nr_samples == 4) &&
          nir_src_is_const(instr->src[0]) &&
          nir_src_as_uint(instr->src[0]) == 0)
         return NULL;

      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          !agx_is_g18g(&b->shader->key->dev) &&
          b->shader->key->fs.nr_samples > 1 &&
          nir_src_is_const(instr->src[0]) &&
          !(nir_src_as_uint(instr->src[0]) & 255))
         return NULL;
      b->shader->out->writes_sample_mask = true;

      /* We need to wait_pix before running Z/S tests, but we don't need to
       * wait_pix before merely discarding. Omit the wait_pix when the affected
       * samples are unconditionally killed.
       */
      /* TARGET and LIVE are independent.  A prolog which only removes the
       * samples excluded by the API mask has disjoint constant sets (for
       * example TARGET=0xfd, LIVE=2 at 2x), so it performs no Z/S tests even
       * though LIVE itself is nonzero.  Treating that as a testing MOVMSK ran
       * depth before the main shader and made a later explicit-early update
       * test the surviving sample twice. */
      bool no_tests = nir_src_is_const(instr->src[1]) &&
         (nir_src_as_uint(instr->src[1]) == 0 ||
          (nir_src_is_const(instr->src[0]) &&
           !(nir_src_as_uint(instr->src[0]) &
             nir_src_as_uint(instr->src[1]))));

       const unsigned g18_samples = b->shader->key->fs.nr_samples;
       if (b->shader->key->dev.emit_agx3d &&
           (g18_samples == 2 || g18_samples == 4) && no_tests) {
           b->shader->out->g18_sample_mask_forms |= AGX_G18_SAMPLE_MASK_KILL;
          /* Native570's G18 multisample discard stages only the affected
           * sample bits: 0xff for whole-pixel discard, or a shifting
           * one-bit low half and zero high half in a memory sample loop.
           * This kills coverage without running depth/stencil tests.
           * Preserve zero for untargeted lanes and leave any later testing
           * update responsible for its own class-1 transaction. */
          agx_index affected = agx_and(b, agx_src_index(&instr->src[0]),
                                      agx_immediate(0xff));
          return agx_sample_mask_agx3(
             b, agx_immediate(1),
             agx_g18_sample_mask_word(b, affected, agx_immediate(0), true), 0);
       }

       /* Native G18 sample-mask writes at 2x/4x stage SR39 in the
       * low half and keep<<8 in the high half, then update under class1.
       * Its fields are bytes, unlike G17P's nibble-based coverage word.
       * Native dynamic-mask fragments use the same word and transaction.
       * Admit all-target late updates and the structurally proven choice
       * between full testing and a non-testing kill. Select the coverage
       * word without changing EXEC around the pixel transaction entry.
       * Early tests and partial-target testing keep their own requirements. */
      if (b->shader->key->dev.emit_agx3d &&
          (g18_samples == 2 || g18_samples == 4) &&
          !b->shader->key->fs.inside_sample_loop &&
          !b->shader->force_early_tests_pending && !no_tests &&
          (agx_g18_all_sample_targets(instr->src[0]) ||
           nir_intrinsic_flags(instr) == AGX_SAMPLE_MASK_FULL_OR_KILL)) {
         b->shader->out->g18_sample_mask_forms |= AGX_G18_SAMPLE_MASK_FULL;
         agx_index coverage = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_coverage_to(b, coverage, AGX_SR_INPUT_SAMPLE_MASK);
         agx_index keep;
         if (nir_src_is_const(instr->src[1])) {
            /* Match the dynamic path: only the low byte contains samples. */
            keep = agx_mov_imm(b, 16,
                               (nir_src_as_uint(instr->src[1]) & 0xffu) << 8);
         } else {
            keep = agx_and(b, agx_src_index(&instr->src[1]),
                           agx_immediate(0xff));
            keep = agx_iadd(b, agx_immediate(0), keep, 4);
            keep = agx_iadd(b, agx_immediate(0), keep, 4);
         }
         agx_index word = agx_temp(b->shader, AGX_SIZE_32);
         agx_emit_collect_to(b, word, 2, (agx_index[]){coverage, keep});
         if (nir_intrinsic_flags(instr) == AGX_SAMPLE_MASK_FULL_OR_KILL) {
            agx_index target = agx_and(b, agx_src_index(&instr->src[0]),
                                       agx_immediate(0xff));
            agx_index tested = agx_and(b, target,
                                       agx_src_index(&instr->src[1]));
            agx_index kill =
               agx_g18_sample_mask_word(b, target, agx_immediate(0), true);
            /* With testing disabled TARGET/LIVE are disjoint. If testing
             * is enabled but LIVE is empty, killing all targets is also
             * equivalent. Zero TARGET produces the native no-op word.
             * Class1 remains balanced for either selected word, as on the
             * existing dynamic one-sample coverage path. */
            agx_index selected = agx_temp(b->shader, AGX_SIZE_32);
            agx_icmpsel_to(b, selected, tested, agx_immediate(0), kill, word,
                           AGX_ICOND_UEQ);
            word = selected;
         }
         return agx_sample_mask_agx3(b, agx_mov_imm(b, 16, 1), word, 2);
      }

      /* Native1121 psdiscard (Apple M6, verified capture 1121, run 1128):
       * Apple's per-sample discard fragment at 2x and 4x acquires class 1
       * once after the transaction entry (pixwait512, pixwait1) and then,
       * for each sample, kills it if discarded -- the staged word {sample
       * bit, 0} at staging 0 with movmsk 160123000000, the Native570 kill
       * word -- and updates it -- the word {(SR39 & 0xff & bit) |
       * (SR39 & 0xff00), samples << 8} at staging 2 with movmsk
       * 160325200000 -- holding class 1 until the colour store's final
       * pixrel524. Mesa's unrolled sample shading proves each per-sample
       * update to be sample_mask((testing | killed) & bit, killed ^ 0xff)
       * (agx_nir_mark_g18_sample_mask_tests, SUBSET flag). Emit those two
       * words for it: the kill word carries TARGET & ~LIVE, the update word
       * TARGET & LIVE, so a part that does not test stages the zero no-op
       * byte. Only single-sample subsets at top level take this form. */
      if (b->shader->key->dev.emit_agx3d &&
          (g18_samples == 2 || g18_samples == 4) &&
          !b->shader->key->fs.inside_sample_loop &&
          !b->shader->force_early_tests_pending && !no_tests &&
          nir_intrinsic_flags(instr) == AGX_SAMPLE_MASK_SUBSET_FULL_OR_KILL &&
          agx_g18_single_sample_target(instr, g18_samples)) {
         if (!(b->shader->out->g18_sample_mask_forms &
               AGX_G18_SAMPLE_MASK_SAMPLE))
            agx_wait_pixel_mask(b, 0x1);
         b->shader->out->g18_sample_mask_forms |= AGX_G18_SAMPLE_MASK_SAMPLE;
         agx_index target = agx_src_index(&instr->src[0]);
         agx_index live = agx_src_index(&instr->src[1]);
         agx_index killed =
            agx_and(b, target, agx_xor(b, live, agx_immediate(0xff)));
         agx_sample_mask_agx3(
            b, agx_immediate(1),
            agx_g18_sample_mask_word(b, killed, agx_immediate(0), true), 0);
         agx_index coverage = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_coverage_to(b, coverage, AGX_SR_INPUT_SAMPLE_MASK);
         agx_index tested = agx_and(b, coverage, agx_immediate(0xff));
         tested = agx_and(b, tested, agx_and(b, target, live));
         agx_index api = agx_and(b, coverage, agx_mov_imm(b, 16, 0xff00));
         agx_index low = agx_iadd(b, tested, api, 0);
         agx_index high =
            agx_mov_imm(b, 16, BITFIELD_MASK(g18_samples) << 8);
         agx_index word = agx_temp(b->shader, AGX_SIZE_32);
         agx_emit_collect_to(b, word, 2, (agx_index[]){low, high});
         return agx_sample_mask_agx3(b, agx_immediate(1), word,
                                     AGX_SAMPLE_MASK_AGX3_SECOND_STAGING);
      }

      if (b->shader->key->dev.emit_agx3d &&
          (g18_samples == 2 || g18_samples == 4))
         b->shader->out->g18_sample_mask_forms |= AGX_G18_SAMPLE_MASK_OTHER;

      /* G17G kills with the same MOVMSK protocol (Apple's applegpu_g17g
       * discard fragment, test/native/g18g oracle.discard.fragment.g17g:
       * stage the word, wait on class 1, MOVMSK, release), and reads the
       * one-sample coverage from the same SR 2. G18G takes it only at one
       * sample: its 2x/4x updates take the native forms above. */
      const bool movmsk_protocol =
         b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
         !(agx_is_g18g(&b->shader->key->dev) &&
           b->shader->key->fs.nr_samples > 1);
      if (movmsk_protocol && !b->shader->key->fs.inside_sample_loop) {
         /* The all/all update inserted at the start of an explicit-early
          * shader is a distinct native protocol: acquire class 1 before
          * reading SR39 and building the coverage word, then keep the class
          * until the fragment's final pixel release.  Ordinary late/sample
          * mask writes instead build and stage the word before acquiring it.
          */
         const bool force_early =
            !no_tests && b->shader->force_early_tests_pending;
         b->shader->force_early_tests_pending = false;

         if (force_early)
            agx_wait_pixel_mask(b, 0x1);

         /* MOVMSK takes one staged coverage word. Native
          * early-fragment-tests code uses word zero for a no-op. Keep TARGET
          * dynamic and select that word without changing EXEC around a
          * coverage operation. */
         const unsigned samples = MAX2(b->shader->key->fs.nr_samples, 1);
         const unsigned all = BITFIELD_MASK(samples);
         /* AGX3_ZS_KILL_ALL_SAMPLES (0xffffff00) is Apple's 1x kill word.
          * G17P uses it for the all-kill update, but staged at an AGX3C
          * multisample rate it does not clear SR39: the epilog's per-sample
          * stores still see every sample covered and a zero API sample mask
          * writes the full colour. Let AGX3C publish the killed byte for
          * every sample through the general path instead. */
         if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3C &&
             samples > 1 && no_tests && nir_src_is_const(instr->src[0]) &&
             nir_src_is_const(instr->src[1]) &&
             (nir_src_as_uint(instr->src[0]) & all) == all &&
             nir_src_as_uint(instr->src[1]) == 0) {
            return agx_sample_mask_agx3(
               b, agx_mov_imm(b, 16, 1),
               agx_mov_imm(b, 32, UINT32_C(0xffffff00)), 0);
         }
         agx_index target_in = agx_src_index(&instr->src[0]);
         agx_index live_in = agx_src_index(&instr->src[1]);
         agx_index target, keep, mid;

         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
             samples > 1) {
            /* Native G17G 4x/8x mask0 publishes the killed byte directly.
             * Mask1/all/variable retain the complete W16 SR39 word in the
             * low half and publish the live byte at bits 24..31. The native
             * AND255 + ANDC255 + ADD reconstructs SR39, not two nibbles.
             * The byte ABI is independent of the raster sample count. */
            const bool constant_target = nir_src_is_const(instr->src[0]);
            const unsigned target_value = constant_target
               ? nir_src_as_uint(instr->src[0]) & 255 : 0;
            agx_index affected = constant_target
               ? agx_mov_imm(b, 16, target_value)
               : agx_and(b, target_in, agx_immediate(255));
            agx_index selected = agx_temp(b->shader, AGX_SIZE_32);
            if (no_tests) {
               agx_mov_to(b, selected, affected);
            } else {
               agx_index live;
               if (constant_target && target_value == 255) {
                  live = agx_and(b, live_in, agx_immediate(255));
               } else {
                  agx_index untargeted = constant_target
                     ? agx_immediate(target_value ^ 255)
                     : agx_xor(b, affected, agx_immediate(255));
                  live = agx_and(b, agx_or(b, untargeted, live_in),
                                 agx_immediate(255));
               }
               agx_index coverage = agx_temp(b->shader, AGX_SIZE_16);
               agx_get_sr_coverage_to(b, coverage, AGX_SR_INPUT_SAMPLE_MASK);
               agx_index high;
               if (constant_target && nir_src_is_const(instr->src[1])) {
                  const unsigned live_value =
                     ((target_value ^ 255) | nir_src_as_uint(instr->src[1])) & 255;
                  high = agx_mov_imm(b, 16, live_value << 8);
               } else {
                  high = agx_iadd(b, agx_zero(),
                     agx_iadd(b, agx_zero(), live, 4), 4);
               }
               agx_index word = agx_temp(b->shader, AGX_SIZE_32);
               agx_emit_collect_to(b, word, 2,
                                  (agx_index[]){coverage, high});
               if (!(constant_target && nir_src_is_const(instr->src[1]))) {
                  /* The testing word carries the whole SR39 coverage, so it
                   * tests every covered sample, and naming a targeted sample
                   * dead in its live byte does not keep that sample from
                   * being tested and written. A per-sample alpha-to-coverage
                   * epilog at alpha 0 wrote depth on M5
                   * (pipeline.*.multisample.alpha_to_coverage.samples_N.
                   * alpha_invisible_check_depth) once the per-lane split no
                   * longer routed such updates to its kill arm. When no
                   * targeted sample is live, publish the killed byte without
                   * testing instead, as that arm and a constant kill do. */
                  agx_index dead = agx_temp(b->shader, AGX_SIZE_32);
                  agx_mov_to(b, dead, agx_and(b, live_in, affected));
                  agx_index killed = agx_temp(b->shader, AGX_SIZE_32);
                  agx_mov_to(b, killed, affected);
                  agx_index tested = agx_temp(b->shader, AGX_SIZE_32);
                  agx_icmpsel_to(b, tested, dead, agx_zero(), killed, word,
                                 AGX_ICOND_UEQ);
                  word = tested;
               }
               if (constant_target) {
                  agx_mov_to(b, selected, word);
               } else {
                  agx_index gate = agx_temp(b->shader, AGX_SIZE_32);
                  agx_mov_to(b, gate, affected);
                  agx_icmpsel_to(b, selected, gate, agx_zero(), word,
                                 agx_zero(), AGX_ICOND_UGT);
               }
            }
            return agx_sample_mask_agx3(b, affected, selected,
               (!no_tests ? 2 : 0) | (force_early ? 1 : 0));
         }

         /* G18G keeps its own one-sample word (agx_g18_sample_mask_word in
          * the shared path below). */
         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
             !agx_is_g18g(&b->shader->key->dev) && samples == 1) {
            /* G17G MOVMSK is not the G17P nibble layout. Native mask0
             * publishes its kill bits directly. Mask1 and a variable mask
             * duplicate SR2 into the low two bytes and publish the live mask
             * in byte3 (metal-mask209); DEPTHF uses the same byte layout. */
            /* agx_m5_discard_word() gives LIVE as the high half (bit 0
             * at bit 8) beside a constant target naming the sample. */
            const bool high_live =
               nir_intrinsic_flags(instr) & BITFIELD_BIT(2);
            const bool whole_target = nir_src_is_const(instr->src[0]) &&
               (nir_src_as_uint(instr->src[0]) & 1);
            agx_index affected = whole_target
               ? agx_mov_imm(b, 16, 1)
               : agx_and(b, target_in, agx_immediate(1));
            agx_index selected = agx_temp(b->shader, AGX_SIZE_32);
            if (no_tests) {
               agx_mov_to(b, selected, affected);
            } else {
               agx_index sr2 = agx_temp(b->shader, AGX_SIZE_16);
               agx_get_coverage_1x_agx3_to(b, sr2);
               agx_index repeated = agx_iadd(b, sr2,
                  agx_iadd(b, agx_zero(), sr2, 4), 4);
               agx_index high = high_live ? live_in : agx_iadd(b, agx_zero(),
                  agx_iadd(b, agx_zero(),
                           agx_and(b, live_in, agx_immediate(1)), 4), 4);
               agx_index word = agx_temp(b->shader, AGX_SIZE_32);
               agx_emit_collect_to(b, word, 2, (agx_index[]){repeated, high});
               if (whole_target) {
                  /* The selector below is constant: it picks the word. */
                  agx_mov_to(b, selected, word);
               } else {
                  agx_index gate = agx_temp(b->shader, AGX_SIZE_32);
                  agx_mov_to(b, gate, affected);
                  agx_icmpsel_to(b, selected, gate, agx_zero(), word,
                                 agx_zero(), AGX_ICOND_UGT);
               }
            }
            return agx_sample_mask_agx3(b, affected, selected,
               (!no_tests ? 2 : 0) | (force_early ? 1 : 0));
         }

         /* agx_g17p_discard_word() reduced these sources to bit 0 already
          * (FLAGS) at one sample, and the select below is one widening
          * 16x16->32 multiply. */
         const unsigned reduced =
            samples == 1 ? nir_intrinsic_flags(instr) : 0;

         if (samples == 1) {
            /* Apple's one-sample fragments read SR2 and repeat it in both
             * coverage nibbles. */
            target = (reduced & 1) ? agx_g17p_reg16(b, target_in)
                                   : agx_and(b, target_in, agx_immediate(1));
            keep = (reduced & 2) ? agx_g17p_reg16(b, live_in)
                                 : agx_and(b, live_in, agx_immediate(1));
            agx_index t16 = agx_temp(b->shader, AGX_SIZE_16);
            agx_get_coverage_1x_agx3_to(b, t16);
            if (b->shader->key->dev.emit_agx3d) {
               /* A zero affected set is a native no-op. Preserve that
                * distinction so a non-testing shader part does not
                * perform the epilog's depth/stencil tests prematurely. */
               agx_index affected = agx_and(b, t16, target);
               keep = agx_and(b, affected, keep);
               return agx_sample_mask_agx3(
                  b, target,
                  agx_g18_sample_mask_word(b, affected, keep, no_tests),
                  (!no_tests ? 2 : 0) | (force_early ? 1 : 0));
            }
            mid = agx_iadd(b, t16, t16, 4);
         } else {
            /* At two and four samples Apple reads SR39, whose low byte holds
             * the covered samples and the API input mask, masks the coverage
             * nibble and recombines them (metal-nt -arch applegpu_g17p,
             * [[sample_mask]] fragments at raster_sample_count 2 and 4).
             * Untargeted samples keep their state, and a word targeting no
             * sample is the no-op, selected by (target + all) >> samples. */
            agx_index masked = agx_and(b, target_in, agx_immediate(all));
            agx_index untargeted =
               agx_xor(b, masked, agx_immediate(all));
            keep = agx_and(b, agx_or(b, untargeted, live_in),
                           agx_immediate(all));
            target = agx_ushr(b, agx_iadd(b, masked, agx_immediate(all), 0),
                              agx_immediate(samples));
            agx_index t16 = agx_temp(b->shader, AGX_SIZE_16);
            agx_get_sr_coverage_to(b, t16, AGX_SR_INPUT_SAMPLE_MASK);
            agx_index upper = agx_ushr(b, t16, agx_immediate(4));
            agx_index covered = agx_and(b, t16, agx_immediate(15));
            /* MOVMSK may only remove samples from the current coverage.  A
             * previous prolog/discard can already have cleared bits in SR39;
             * using the requested keep set directly resurrects those samples
             * and makes a later all-kill update leave one sample live.  This
             * is the second BAND in Apple's native conditional-kill prefix. */
            keep = agx_and(b, keep, covered);
            /* Native MOVMSK has the API mask in bits 4..7 and coverage in
             * bits 8..11, like DEPTHF. Preserve that order after a previous
             * update has made the two SR39 nibbles different. */
            mid = agx_iadd(b, upper, covered, 4);
         }

         agx_index word = agx_iadd(b, keep, mid, 4);
         agx_index selected = agx_temp(b->shader, AGX_SIZE_32);
         /* target is 0 or 1 and word < 0x1000, so the zero-extended 16x16->32
          * product (Apple's half-operand imadd form) is the selected word,
          * high half zero. */
         agx_imad_to(b, selected, agx_abs(target), agx_abs(word), agx_zero(),
                     0);
         return agx_sample_mask_agx3(
            b, target, selected,
            (!no_tests ? 2 : 0) | (force_early ? 1 : 0));
      }

      if (!no_tests)
         agx_wait_pixel_mask(b, 0x1);

      return agx_sample_mask(b, agx_src_index(&instr->src[0]),
                             agx_src_index(&instr->src[1]));
   }

   case nir_intrinsic_load_back_face_agx:
      return agx_get_sr_to(b, dst, AGX_SR_BACKFACING);

   case nir_intrinsic_load_samples_log2_agx:
      return agx_get_sr_to(b, dst, AGX_SR_SAMPLES_LOG2);

   case nir_intrinsic_load_sample_mask_in:
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         /* Apple's G17P fragment ABI exposes the API-visible sample mask in
          * bits 4..7 of native W16 SR39.  Extract it before widening to NIR's
          * uint32_t result; reading SR39 as W32 selects an unwitnessed form.
          * G17G and G18G keep it in bits 8..15 of the same word: their
          * [[sample_mask]] fragments read SR39 and shift it right by 8 at
          * 1x, 2x and 4x alike (metal-nt applegpu_g17g/applegpu_g18g,
          * m6-sample-mask-in). */
         const unsigned shift =
            b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C ? 8 : 4;
         agx_index native = agx_temp(b->shader, AGX_SIZE_16);
         agx_index shifted = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_to(b, native, AGX_SR_INPUT_SAMPLE_MASK);
         agx_ushr_to(b, shifted, native, agx_immediate(shift));
         return agx_mov_to(b, dst, shifted);
      }
      return agx_get_sr_to(b, dst, AGX_SR_INPUT_SAMPLE_MASK);

   case nir_intrinsic_load_sample_mask:
      if ((b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
           !agx_is_g18g(&b->shader->key->dev)) ||
          (agx_is_g18g(&b->shader->key->dev) &&
           (b->shader->key->fs.nr_samples == 2 ||
            b->shader->key->fs.nr_samples == 4))) {
         /* G17P/G18G covered samples are bits 0..3 of native W16 SR39, the word
          * Apple's multisample fragments AND with each sample's bit before
          * that sample's tile store. Unlike the API input mask, this changes
          * after depth/stencil tests and cannot be CSE'd across them. Generic
          * MS image spills also consume it; AGX2's W32 SR60 is not their G18
          * coverage register. Native454 grounds the G18 sample-word use. */
         agx_index native = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_coverage_to(b, native, AGX_SR_INPUT_SAMPLE_MASK);
         return agx_mov_to(b, dst, agx_and(b, native, agx_immediate(
            b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C ? 255 : 15)));
      }
      return agx_get_sr_coverage_to(b, dst, AGX_SR_COVERAGE_MASK);

   case nir_intrinsic_load_active_samples_agx:
      /* Everywhere else this is a loop variable that the MSAA lowerings have
       * already substituted. It survives to the backend only on G17P, where
       * agx_nir_lower_g17p_tile_samples asks for the live coverage word as a
       * real register operand for lstcol/lldcol: Apple's own multisample
       * fragments read exactly this, native W16 SR39, and pass it straight to
       * the tile access. Bits 0..3 are the covered samples -- the per-sample
       * form ANDs this same word with 1/2/4/8. Use the ordered coverage read
       * so scheduling and CSE cannot reuse pre-depth coverage for tile stores.
       *
       * msaa163: G18G's four-sample tile lowering (agx_nir_lower_g18_tile_
       * samples) leaves it too. Apple's G18 4x fragment reads
       * the same W16 SR39 into a half and passes it unmodified to lstcol
       * (rld 210a42000000: f21=39, f10/f24=1). */
      assert(((b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
               !agx_is_g18g(&b->shader->key->dev)) ||
              (agx_is_g18g(&b->shader->key->dev) &&
               (b->shader->key->fs.nr_samples == 2 ||
                b->shader->key->fs.nr_samples == 4))) &&
             "only the G17P/G18G tile lowerings leave this for the backend");
      assert(instr->def.bit_size == 16 && "16-bit sample mask");
      return agx_get_sr_coverage_to(b, dst, AGX_SR_INPUT_SAMPLE_MASK);

   case nir_intrinsic_load_helper_invocation:
      /* Compare special register to zero. We could lower this in NIR (letting
       * us fold in an inot) but meh?
       */
      return agx_icmp_to(b, dst,
                         agx_get_sr_coverage(
                            b, b->shader->key->dev.encoding ==
                                  AGX_ISA_ENCODING_AGX2 ? 32 : 16,
                            AGX_SR_IS_ACTIVE_THREAD),
                         agx_zero(), AGX_ICOND_UEQ, false);

   case nir_intrinsic_load_vertex_id_zero_base:
      assert(stage == MESA_SHADER_COMPUTE && "only for SW VS");
      return agx_mov_to(b, dst, agx_abs(agx_vertex_id_zero_base(b)));

   case nir_intrinsic_load_vertex_id:
      /* AGX3 hardware vertex shaders read the invocation index explicitly.
       * Apple's isolated G17P vertex-attribute shaders begin with
       * `0cdd1006`, a 32-bit rld from SR 0xdd. Retaining AGX2's preloaded r5
       * ABI made every invocation observe the same index and collapse a
       * triangle to one vertex. Software vertex shaders still use the preload
       * ABI populated by their compute launcher. */
      if (stage == MESA_SHADER_VERTEX &&
          b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
         return agx_get_sr_to(b, dst, AGX_SR_VERTEX_ID);

      return agx_mov_to(b, dst, agx_abs(agx_vertex_id(b)));

   case nir_intrinsic_load_primitive_id:
      /* AGX3's tessellator hands the patch index to the post-tessellation
       * vertex program in SR 0xd9, Apple's [[patch_id]]. */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         assert(stage == MESA_SHADER_VERTEX && "post-tessellation vertex");
         return agx_get_sr_to(b, dst, AGX_SR_TESS_PATCH_ID);
      }

      assert(stage == MESA_SHADER_TESS_EVAL);
      return agx_mov_to(b, dst, agx_abs(agx_vertex_id(b)));

   case nir_intrinsic_load_instance_id:
      /* The matching Apple hardware-vertex witness is `0cd81006`, reading
       * instance ID from SR 0xd8. */
      if (stage == MESA_SHADER_VERTEX &&
          b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
         return agx_get_sr_to(b, dst, AGX_SR_INSTANCE_ID);

      return agx_mov_to(b, dst, agx_abs(agx_instance_id(b)));

   case nir_intrinsic_load_tess_coord_xy: {
      /* On AGX3 the tessellator's position in the patch arrives as two
       * 32-bit float special registers, Apple's [[position_in_patch]]. */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         assert(stage == MESA_SHADER_VERTEX && "post-tessellation vertex");
         agx_index coords[] = {agx_get_sr(b, 32, AGX_SR_TESS_COORD_U),
                               agx_get_sr(b, 32, AGX_SR_TESS_COORD_V)};
         return agx_emit_collect_to(b, dst, 2, coords);
      }

      assert(stage == MESA_SHADER_TESS_EVAL);

      agx_index coords[] = {agx_tess_coord_x(b), agx_tess_coord_y(b)};
      return agx_emit_collect_to(b, dst, 2, coords);
   }

   case nir_intrinsic_load_preamble:
      return agx_emit_load_preamble(b, dst, instr);

   case nir_intrinsic_store_preamble:
      return agx_emit_store_preamble(b, instr);

   case nir_intrinsic_image_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
      return agx_emit_image_load(b, dst, instr);

   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_store:
      return agx_emit_image_store(b, instr);

   case nir_intrinsic_image_store_block_agx:
   case nir_intrinsic_bindless_image_store_block_agx:
      return agx_emit_block_image_store(b, instr);

   case nir_intrinsic_load_workgroup_id:
      return agx_load_compute_dimension(b, dst, instr,
                                        AGX_SR_THREADGROUP_POSITION_IN_GRID_X);

   case nir_intrinsic_load_workgroup_size:
      return agx_load_compute_dimension(b, dst, instr,
                                        AGX_SR_THREADS_PER_THREADGROUP_X);

   case nir_intrinsic_load_global_invocation_id:
      return agx_load_compute_dimension(b, dst, instr,
                                        AGX_SR_THREAD_POSITION_IN_GRID_X);

   case nir_intrinsic_load_local_invocation_id:
      return agx_load_compute_dimension(
         b, dst, instr, AGX_SR_THREAD_POSITION_IN_THREADGROUP_X);

   case nir_intrinsic_load_scs_thread_index_agx:
      return agx_get_sr_native_width_to(b, dst, AGX_SR_SCS_THREAD_INDEX);

   case nir_intrinsic_load_local_invocation_index:
      return agx_get_sr_native_width_to(
         b, dst, AGX_SR_THREAD_INDEX_IN_THREADGROUP);

   case nir_intrinsic_load_layer_id:
      /* Driver-internal fragment programs read the tile's own layer rather
       * than an API gl_Layer. G17P's native SR 0xc3 and G18G's native array
       * EOT programs use the same normalized SR12, W16:
       *
       *  - the builtin background reload (fs.background), and
       *  - with a spilled render target, the fragment epilog's eMRT image
       *    store, whose layer coordinate image_coords() in
       *    agx_nir_lower_tilebuffer.c builds from this intrinsic. Reading the
       *    compute grid-Z input there made every layered eMRT store address
       *    layer zero.
       *
       * Both are driver-internal, and hk sets g17p_emrt only for the eMRT
       * epilog (hk_cmd_draw.c). An API gl_Layer read never reaches here from
       * hk -- nir_lower_sysvals_to_varyings rewrites it as a flat coefficient
       * first (hk_shader.c) -- but it does from gallium, which lowers
       * VARYING_SLOT_LAYER back to this intrinsic with layer_as_varying
       * false and sets neither key bit. Leave that path alone. */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
          stage == MESA_SHADER_FRAGMENT &&
          (b->shader->key->fs.background ||
           b->shader->key->fs.ignore_tib_dependencies ||
           b->shader->key->g17p_emrt))
         return agx_get_sr_native_width_to(b, dst, AGX_SR_BACKGROUND_LAYER_ID);
      return agx_get_sr_to(b, dst, AGX_SR_THREADGROUP_POSITION_IN_GRID_Z);

   case nir_intrinsic_barrier: {
      assert(!b->shader->is_preamble && "invalid");

      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         bool emitted = false;

         if (nir_intrinsic_memory_scope(instr) != SCOPE_NONE) {
            const nir_variable_mode modes =
               nir_intrinsic_memory_modes(instr);
            const bool memory_only =
               nir_intrinsic_execution_scope(instr) < SCOPE_WORKGROUP;
            const bool memory_only_g17p =
               b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
               memory_only;
            /* G17G: atomic_thread_fence(mem_texture) is the threadgroup
             * fence pair at threadgroup scope and texture.fence()'s pair at
             * device and simdgroup scope; threadgroup_barrier(mem_texture)
             * is the rendezvous pair (port/fencewit). */
            const bool memory_only_m5 =
               b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
               memory_only;
            const bool device_rendezvous =
               (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
                b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
               !memory_only &&
               nir_intrinsic_memory_scope(instr) >= SCOPE_QUEUE_FAMILY;

            /* The AGX2 cache-operation train collapses on AGX3 into one
             * exact operation per semantic memory scope. G17P distinguishes
             * a memory-only fence from a threadgroup execution barrier with
             * the same memory class; choose from NIR's execution scope. Keep
             * Apple's observed device -> texture -> threadgroup ordering. A tessellation
             * control shader runs as a compute kernel whose outputs poly
             * lowered to the global output buffer, so its output barrier is
             * a device-memory barrier. */
            const bool tcs_outputs =
               b->shader->nir->info.stage == MESA_SHADER_TESS_CTRL &&
               (modes & nir_var_shader_out);
            if ((modes & (nir_var_mem_global | nir_var_image)) ||
                tcs_outputs) {
               agx_scoped_barrier_agx3(
                  b, nir_intrinsic_execution_scope(instr) < SCOPE_WORKGROUP
                        ? AGX3_BARRIER_SCOPE_DEVICE_FENCE
                        : AGX3_BARRIER_SCOPE_DEVICE);

               /* On G17P and M5 the workgroup barrier's device-memory form
                * orders device memory only within the workgroup (the other
                * AGX3B chips are untested and take the same fence): writes
                * before it can stay invisible to other workgroups after it,
                * and a
                * plain load after it can still return a line this core
                * cached before another workgroup wrote it. At device or
                * queue-family memory scope, follow the rendezvous with the
                * device fence, so that it comes after every write the
                * workgroup made before the barrier and before every load
                * and signal after it.
                */
               if (device_rendezvous)
                  agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_DEVICE_FENCE);
               emitted = true;
            }

            if (modes & nir_var_image) {
               const bool wait_for_prior_texture =
                  nir_intrinsic_memory_semantics(instr) &
                  (NIR_MEMORY_RELEASE | NIR_MEMORY_MAKE_AVAILABLE);
               agx_scoped_barrier_agx3(
                  b, memory_only_g17p
                        ? (wait_for_prior_texture
                              ? AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT
                              : AGX3_BARRIER_SCOPE_TEXTURE_FENCE)
                     : memory_only_m5
                        ? (nir_intrinsic_memory_scope(instr) == SCOPE_WORKGROUP
                              ? AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP
                              : AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT)
                        : AGX3_BARRIER_SCOPE_TEXTURE);
               emitted = true;
            }

            if (modes & nir_var_mem_shared) {
               const bool subgroup =
                  nir_intrinsic_execution_scope(instr) <= SCOPE_SUBGROUP &&
                  nir_intrinsic_memory_scope(instr) <= SCOPE_SUBGROUP;
               agx_scoped_barrier_agx3(
                  b, subgroup ? AGX3_BARRIER_SCOPE_SUBGROUP_SHARED
                              : AGX3_BARRIER_SCOPE_THREADGROUP);
               emitted = true;
            }
         }

         /* A workgroup execution barrier with no memory scope is the
          * recovered mem_none form. Barriers selected above for an actual
          * workgroup execution scope already carry the rendezvous. */
         if (nir_intrinsic_execution_scope(instr) >= SCOPE_WORKGROUP &&
             !emitted) {
            assert(mesa_shader_stage_is_compute(b->shader->nir->info.stage));
            agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_EXECUTION);
         }

         return NULL;
      }

      bool needs_image_barriers = false;

      if (nir_intrinsic_memory_scope(instr) != SCOPE_NONE) {
         nir_variable_mode modes = nir_intrinsic_memory_modes(instr);

         if (modes & (nir_var_mem_global | nir_var_image)) {
            agx_memory_barrier(b);

            /* Pull out all the big hammers to make cross-workgroup memory
             * barriers work.
             */
            if (nir_intrinsic_memory_scope(instr) >= SCOPE_QUEUE_FAMILY) {
               agx_unknown_barrier_1(b);
               agx_memory_barrier_2(b);

               /* These are observed on G13D. At some point we should figure out
                * what the individual opcodes do.
                */
               agx_device_barrier_2(b);
               agx_unknown_barrier_2(b);
               agx_memory_barrier_3(b);
               agx_device_barrier_1(b);
            }
         }

         if (modes & nir_var_image) {
            agx_image_barrier_1(b);
            agx_image_barrier_2(b);
            needs_image_barriers = true;
         }
      }

      /* Nothing to do for subgroup barriers */
      if (nir_intrinsic_execution_scope(instr) >= SCOPE_WORKGROUP) {
         assert(mesa_shader_stage_is_compute(b->shader->nir->info.stage));

         agx_threadgroup_barrier(b);
      }

      if (needs_image_barriers) {
         agx_image_barrier_3(b);
         agx_image_barrier_4(b);

         /* Image atomics are lowered to global memory, while imageLoad uses
          * the texture cache. After acquiring another invocation's atomic
          * payload, retire the old texture-cache contents just as
          * fence_mem_to_tex_agx does for a local image atomic. The image/PBE
          * barrier train alone does not include this memory-to-texture step.
          * A release-only barrier has no subsequent reads to make visible.
          */
         if (nir_intrinsic_memory_semantics(instr) &
             (NIR_MEMORY_ACQUIRE | NIR_MEMORY_MAKE_VISIBLE))
            agx_flush_memory_to_texture(b);
      }

      return NULL;
   }

   case nir_intrinsic_fence_pbe_to_tex_agx: {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         /* texture.fence(): on AGX3C it waits on the writes' own slots. */
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT);
         return NULL;
      }

      agx_image_barrier_1(b);
      agx_image_barrier_2(b);
      agx_image_barrier_3(b);
      agx_image_barrier_4(b);
      return NULL;
   }

   case nir_intrinsic_fence_mem_to_tex_agx: {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_DEVICE);
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE);
         return NULL;
      }

      /* Flush out the atomic to main memory... Found experimentally... */
      agx_memory_barrier(b);
      agx_memory_barrier_2(b);

      /* TODO: Which ones do we actually need? */
      agx_image_barrier_1(b);
      agx_image_barrier_2(b);
      agx_image_barrier_3(b);
      agx_image_barrier_4(b);

      /* Flush out the texture cache */
      agx_flush_memory_to_texture(b);
      return NULL;
   }

   case nir_intrinsic_fence_pbe_to_tex_pixel_agx: {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE);
         return NULL;
      }

      agx_image_barrier_1(b);
      agx_image_barrier_2(b);
      agx_flush_memory_to_texture(b);
      agx_image_barrier_3(b);
      return NULL;
   }

   case nir_intrinsic_fence_helper_exit_agx: {
      assert(b->shader->key->is_helper);
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_DEVICE);
         return NULL;
      }

      agx_memory_barrier(b);
      agx_unknown_barrier_1(b);
      agx_memory_barrier_2(b);
      agx_unknown_barrier_2(b);
      agx_memory_barrier_3(b);
      return NULL;
   }

   case nir_intrinsic_begin_invocation_interlock: {
      b->shader->any_interlock = true;
      agx_wait_pixel_mask(b, 0xC);
      return NULL;
   }

   case nir_intrinsic_end_invocation_interlock:
      /* The end of the fragment releases class 12, which only lengthens the
       * critical section. */
      return NULL;

   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddx_fine:
      return agx_dfdx_to(b, dst, agx_src_index(&instr->src[0]));

   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_coarse:
   case nir_intrinsic_ddy_fine:
      return agx_dfdy_to(b, dst, agx_src_index(&instr->src[0]));

   case nir_intrinsic_load_subgroup_invocation:
      return agx_get_sr_native_width_to(b, dst,
                                        AGX_SR_THREAD_INDEX_IN_SUBGROUP);

   case nir_intrinsic_object_payload_exit_barrier_agx:
      return agx_scoped_barrier_agx3(b,AGX3_BARRIER_SCOPE_OBJECT_PAYLOAD_EXIT);
   case nir_intrinsic_mesh_payload_entry_barrier_agx:
      return agx_scoped_barrier_agx3(b,AGX3_BARRIER_SCOPE_MESH_PAYLOAD_ENTRY);
   case nir_intrinsic_load_mesh_payload_address_agx: {
      agx_index parts[2]={agx_temp(b->shader,AGX_SIZE_32),
                          agx_temp(b->shader,AGX_SIZE_32)};
      agx_get_sr_native_width_to(b,parts[0],AGX_SR_MESH_PAYLOAD_LO);
      agx_get_sr_native_width_to(b,parts[1],AGX_SR_MESH_PAYLOAD_HI);
      return agx_emit_collect_to(b,dst,2,parts);
   }
   case nir_intrinsic_load_mesh_primitive_storage_agx: {
      agx_index parts[2]={agx_temp(b->shader,AGX_SIZE_32),
                          agx_temp(b->shader,AGX_SIZE_32)};
      agx_get_sr_native_width_to(b,parts[0],AGX_SR_MESH_PRIMITIVE_STORAGE_LO);
      agx_get_sr_native_width_to(b,parts[1],AGX_SR_MESH_PRIMITIVE_STORAGE_HI);
      return agx_emit_collect_to(b,dst,2,parts);
   }
   case nir_intrinsic_allocate_mesh_uvb_agx:
      return agx_allocate_uvb_m5(b, agx_src_index(&instr->src[0]),
                                 agx_src_index(&instr->src[1]));
   case nir_intrinsic_fence_mesh_uvb_agx:
      return agx_fence_uvb_gen1(b, nir_intrinsic_base(instr));
   case nir_intrinsic_store_mesh_uvb_agx:
      return agx_store_uvb_m5(b, agx_src_index(&instr->src[0]),
                              nir_intrinsic_flags(instr));
   case nir_intrinsic_store_object_childcount_agx:
      return agx_store_object_childcount_m5(b, agx_src_index(&instr->src[0]));
   case nir_intrinsic_load_subgroup_id:
      return agx_get_sr_native_width_to(
         b, dst, AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP);

   case nir_intrinsic_load_active_subgroup_invocation_agx:
      return agx_get_sr_coverage_native_width_to(
         b, dst, AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP);

   case nir_intrinsic_load_active_subgroup_count_agx:
      return agx_get_sr_coverage_native_width_to(
         b, dst, AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP);

   case nir_intrinsic_reduce: {
      assert((instr->def.bit_size == 1 || instr->def.bit_size == 16 ||
              instr->def.bit_size == 32) &&
             "should've been lowered");

      unsigned cluster_size = nir_intrinsic_cluster_size(instr);
      assert(cluster_size == 0 || cluster_size == 4 || cluster_size >= 32);

      enum agx_simd_op op =
         translate_simd_op(nir_intrinsic_reduction_op(instr));

      agx_index src0 = agx_src_index(&instr->src[0]);

      if (cluster_size == 4)
         return agx_quad_reduce_to(b, dst, src0, op);
      else
         return agx_simd_reduce_to(b, dst, src0, op);
   }

   case nir_intrinsic_exclusive_scan: {
      assert((instr->def.bit_size == 1 || instr->def.bit_size == 16 ||
              instr->def.bit_size == 32) &&
             "should've been lowered");

      return agx_simd_prefix_to(
         b, dst, agx_src_index(&instr->src[0]),
         translate_simd_op(nir_intrinsic_reduction_op(instr)));
   }

   case nir_intrinsic_read_invocation: {
      /* TODO: Check if we're actually inside divergent control flow */
      b->shader->any_quad_divergent_shuffle |= b->shader->any_cf;

      /* Lane ID guaranteed to be uniform */
      agx_index lane = agx_src_index(&instr->src[1]);

      /*
       * AGX3's immediate-lane broadcast puts the lane in a seven-bit field,
       * so lanes 0..127 -- every lane a subgroup can have -- encode directly.
       * A constant outside that range is undefined behaviour rather than a
       * program we must reject, so materialise it into the register-lane
       * form, which takes any value, instead of refusing the shader.
       */
      if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
          lane.type == AGX_INDEX_IMMEDIATE && lane.value > 0x7f) {
         agx_index reg = agx_temp(b->shader, AGX_SIZE_32);
         agx_mov_imm_to(b, reg, lane.value);
         lane = reg;
      }

      agx_index data = agx_src_index(&instr->src[0]);
      /* G18G's admitted shuffle carries 32-bit values. Preserve the low-half
       * bits through that path, just as for a dynamic quad broadcast. M6 only:
       * the M5 keeps its half shuffle. */
      if (agx_is_g18g(&b->shader->key->dev) &&
          data.size == AGX_SIZE_16 && agx_channels(data) == 1) {
         agx_index shuffled = agx_temp(b->shader, AGX_SIZE_32);
         agx_shuffle_to(b, shuffled, agx_pad_to_32(b, data), lane);
         return agx_subdivide_to(b, dst, shuffled, 0);
      }

      return agx_shuffle_to(b, dst, data, lane);
   }

   case nir_intrinsic_quad_broadcast: {
      /* TODO: Check if we're actually inside divergent control flow */
      b->shader->any_quad_divergent_shuffle |= b->shader->any_cf;

      const agx_index data = agx_src_index(&instr->src[0]);
      const agx_index lane = agx_src_index(&instr->src[1]);

      /*
       * A 16-bit quad shuffle is encodable only from an IMMEDIATE lane. That
       * word narrows its width pair (f10 = f24 = 1) and carries the quad
       * selector in a different field, so the two compose; the register-lane
       * word does not, and agx3_subgroup_encode_shuffle() has no witness for
       * it and refuses. A dynamic lane over 16-bit data therefore had no
       * encoding at all, and the refusal took the whole shader down rather
       * than just the instruction.
       *
       * MEASURED on Neo (G17P, A18 Pro) at agx3 a0b77f09611, running
       * dEQP-VK.subgroups.quad.compute.subgroupquadbroadcast_nonconst_float16_t
       * with AGX_MESA_DEBUG=refuse:
       *
       *   agx3_pack_binary.c:10028: no grounded AGX3 encoding for quad_shuffle
       *     [d16:reg13 s0=16:reg13 s1=32:reg2 ...]
       *   AGX3 refused shader during binary packing
       *   hk: vkCreateComputePipelines refused pipeline 0 of 1: VK_ERROR_UNKNOWN
       *
       * Any quad broadcast whose lane is not a compile-time constant lands
       * here; s1 above is r1, the register agx_lower_divergent_shuffle()
       * reserves for a shuffle index under divergent control flow.
       *
       * Widen to the 32-bit register-lane form, which is exactly the shape
       * nir_lower_subgroups already hands us for 8- and 16-bit VECTORS -- it
       * bitcasts them to 32 bits before scalarising, which is why f16vec2 and
       * i8vec4 pass today while the scalars did not. Booleans and 8-bit
       * values already live in 16-bit registers, so this covers them too.
       * AGX2 is untouched: it has its own packer and its own witnesses.
       */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
          data.size == AGX_SIZE_16 && agx_channels(data) == 1 &&
          lane.type != AGX_INDEX_IMMEDIATE) {
         agx_index shuffled = agx_temp(b->shader, AGX_SIZE_32);
         agx_quad_shuffle_to(b, shuffled, agx_pad_to_32(b, data), lane);
         return agx_subdivide_to(b, dst, shuffled, 0);
      }

      /* Lane ID guaranteed to be uniform */
      return agx_quad_shuffle_to(b, dst, data, lane);
   }

   case nir_intrinsic_shuffle_xor:
      /* Only constant masks reach here, see agx_nir_lower_subgroups_m5() */
      if (!nir_src_is_const(instr->src[1])) {
         b->shader->unsupported_instruction = true;
         return NULL;
      }

      return agx_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                agx_immediate(nir_src_as_uint(instr->src[1])));

   case nir_intrinsic_quad_swap_horizontal: {
      return agx_quad_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                     agx_immediate(1));
   }

   case nir_intrinsic_quad_swap_vertical: {
      return agx_quad_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                     agx_immediate(2));
   }

   case nir_intrinsic_quad_swap_diagonal: {
      return agx_quad_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                     agx_immediate(3));
   }

   case nir_intrinsic_ballot: {
      return agx_ballot_to(b, dst, agx_src_index(&instr->src[0]));
   }

   case nir_intrinsic_quad_ballot_agx: {
      return agx_quad_ballot_to(b, dst, agx_src_index(&instr->src[0]));
   }

   case nir_intrinsic_intersect_ray_gen1_agx:
      return agx_intersect_ray_gen1(b, agx_src_index(&instr->src[0]),
         agx_src_index(&instr->src[1]), agx_src_index(&instr->src[2]),
         agx_src_index(&instr->src[3]), nir_intrinsic_base(instr));
   case nir_intrinsic_continue_ray_gen1_agx:
      return agx_continue_ray_gen1(b, agx_src_index(&instr->src[0]),
                                  nir_intrinsic_base(instr));

   case nir_intrinsic_load_scs_agx: {
      agx_instr *load = agx_scs_load_agx3_to(
         b, dst, nir_intrinsic_base(instr),
         instr->def.bit_size == 16 ? AGX_FORMAT_I16 : AGX_FORMAT_I32,
         BITFIELD_MASK(instr->def.num_components));
      agx_emit_cached_split(b, dst, instr->def.num_components);
      return load;
   }

   case nir_intrinsic_store_scs_agx:
      return agx_scs_store_agx3(b, agx_src_index(&instr->src[0]),
                               nir_intrinsic_base(instr),
                               instr->src[0].ssa->bit_size == 16
                                  ? AGX_FORMAT_I16 : AGX_FORMAT_I32,
                               BITFIELD_MASK(instr->num_components));

   case nir_intrinsic_load_scs_indexed_agx: {
      if (nir_intrinsic_flags(instr) & AGX_NIR_SCS_TOUCH)
         return agx_emit_scs_touch(b, instr, true);
      agx_instr *load = agx_scs_load_register_indexed_agx3_to(
         b, dst, agx_src_index(&instr->src[0]),
         agx_src_index(&instr->src[1]), nir_intrinsic_base(instr),
         instr->def.bit_size == 16 ? AGX_FORMAT_I16 : AGX_FORMAT_I32,
         BITFIELD_MASK(instr->def.num_components),
         nir_intrinsic_component(instr), nir_intrinsic_flags(instr) & 1, false);
      agx_emit_cached_split(b, dst, instr->def.num_components);
      return load;
   }

   case nir_intrinsic_store_scs_indexed_agx:
      return agx_scs_store_register_indexed_agx3(
         b, agx_src_index(&instr->src[0]), agx_src_index(&instr->src[1]),
         agx_src_index(&instr->src[2]), nir_intrinsic_base(instr),
         AGX_FORMAT_I32, BITFIELD_MASK(instr->num_components),
         nir_intrinsic_component(instr));

   case nir_intrinsic_load_scs_register_agx: {
      agx_instr *load = agx_scs_load_register_indexed_agx3_to(
         b, dst, agx_src_index(&instr->src[0]), agx_src_index(&instr->src[1]),
         nir_intrinsic_base(instr), agx_format_for_pipe(nir_intrinsic_format(instr)),
         BITFIELD_MASK(instr->def.num_components), nir_intrinsic_component(instr),
         !!(nir_intrinsic_flags(instr) & 1), !!(nir_intrinsic_flags(instr) & 8));
      agx_emit_cached_split(b, dst, instr->def.num_components);
      return load;
   }

   case nir_intrinsic_load_scs_uniform_agx:
   case nir_intrinsic_load_scs_base_agx: {
      const bool uniform = instr->intrinsic == nir_intrinsic_load_scs_uniform_agx;
      unsigned flags = nir_intrinsic_flags(instr);
      if (!uniform && (flags & AGX_NIR_SCS_TOUCH))
         return agx_emit_scs_touch(b, instr, false);
      enum agx_format fmt = agx_format_for_pipe(nir_intrinsic_format(instr));
      unsigned mask = BITFIELD_MASK(instr->def.num_components);
      agx_instr *load = uniform ? agx_scs_load_uniform_indexed_agx3_to(
         b, dst, agx_src_index(&instr->src[0]), agx_src_index(&instr->src[1]),
         nir_intrinsic_base(instr), fmt, mask, nir_intrinsic_component(instr),
         !!(flags & 2), !!(flags & 1)) :
         agx_scs_load_register_base_agx3_to(
         b, dst, agx_src_index(&instr->src[0]), nir_intrinsic_base(instr),
         fmt, mask, nir_intrinsic_component(instr), !!(flags & 1), !!(flags & 4));
      load->scs_rt_wait = b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B && (flags & 8);
      load->scs_rt_completion = b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C && !uniform && (flags & 8);
      agx_emit_cached_split(b, dst, instr->def.num_components);
      return load;
   }

   case nir_intrinsic_store_scs_uniform_agx:
   case nir_intrinsic_store_scs_base_agx: {
      const bool uniform = instr->intrinsic == nir_intrinsic_store_scs_uniform_agx;
      enum agx_format fmt = agx_format_for_pipe(nir_intrinsic_format(instr));
      unsigned mask = BITFIELD_MASK(instr->num_components);
      return uniform ? agx_scs_store_uniform_indexed_agx3(
         b, agx_src_index(&instr->src[0]), agx_src_index(&instr->src[1]),
         agx_src_index(&instr->src[2]), nir_intrinsic_base(instr), fmt, mask,
         nir_intrinsic_component(instr)) :
         agx_scs_store_register_base_agx3(
         b, agx_src_index(&instr->src[0]), agx_src_index(&instr->src[1]),
         nir_intrinsic_base(instr), fmt, mask, nir_intrinsic_component(instr));
   }

   case nir_intrinsic_intersect_ray_agx:
      return agx_intersect_ray_m5(b, agx_src_index(&instr->src[0]),
                                agx_src_index(&instr->src[1]),
                                agx_src_index(&instr->src[2]),
                                agx_src_index(&instr->src[3]),
                                agx_src_index(&instr->src[4]));

   case nir_intrinsic_enter_query_agx:
      return agx_enter_query_m5_to(b, dst,
                                   nir_intrinsic_base(instr) ?: 4);

   case nir_intrinsic_continue_ray_agx:
      return agx_continue_ray_m5(b, agx_src_index(&instr->src[0]));

   case nir_intrinsic_doorbell_agx: {
      return agx_doorbell(b, nir_src_as_uint(instr->src[0]));
   }

   case nir_intrinsic_stack_map_agx: {
      return agx_stack_map(b, agx_src_index(&instr->src[1]),
                           nir_src_as_uint(instr->src[0]));
   }

   case nir_intrinsic_stack_unmap_agx: {
      return agx_stack_unmap_to(b, dst, nir_src_as_uint(instr->src[0]));
   }

   case nir_intrinsic_load_scratch:
      agx_emit_load_scratch(b, dst, instr);
      return NULL;

   case nir_intrinsic_store_scratch:
      agx_emit_store_scratch(b, instr);
      return NULL;

   case nir_intrinsic_load_core_id:
      return agx_get_sr_to(b, dst, AGX_SR_CORE_ID);

   case nir_intrinsic_load_helper_op_id_agx:
      assert(b->shader->key->is_helper);
      return agx_get_sr_barrier_to(b, dst, AGX_SR_HELPER_OP);

   case nir_intrinsic_load_helper_arg_lo_agx:
      assert(b->shader->key->is_helper);
      return agx_get_sr_barrier_to(b, dst, AGX_SR_HELPER_ARG_L);

   case nir_intrinsic_load_helper_arg_hi_agx:
      assert(b->shader->key->is_helper);
      return agx_get_sr_barrier_to(b, dst, AGX_SR_HELPER_ARG_H);

   case nir_intrinsic_load_exported_agx:
      return agx_load_exported_to(b, dst, nir_intrinsic_base(instr),
                                  instr->def.num_components);

   case nir_intrinsic_export_agx:
      return agx_emit_export(b, nir_intrinsic_base(instr), instr->src[0]);

   case nir_intrinsic_bindless_image_agx:
   case nir_intrinsic_bindless_sampler_agx:
      /* These must always be chased */
      return NULL;

   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_sample_id:
   case nir_intrinsic_load_sample_pos:
      UNREACHABLE("Sample shading should have been lowered");

   default:
      /* For instance per-primitive fragment inputs, which only mesh
       * shading produces. Refuse the shader rather than run off the end
       * of this switch, which is undefined behaviour in release builds. */
      fprintf(stderr, "Unhandled intrinsic %s\n",
              nir_intrinsic_infos[instr->intrinsic].name);
      b->shader->unsupported_instruction = true;
      return NULL;
   }
}

static agx_index
agx_alu_src_index(agx_builder *b, nir_alu_src src)
{
   /* Check well-formedness of the input NIR */
   ASSERTED unsigned bitsize = nir_src_bit_size(src.src);
   unsigned comps = nir_src_num_components(src.src);
   unsigned channel = src.swizzle[0];

   assert(bitsize == 1 || bitsize == 8 || bitsize == 16 || bitsize == 32 ||
          bitsize == 64);
   assert(channel < comps);

   return agx_extract_nir_src(b, src.src, channel);
}

/*
 * Emit an instruction translating (s0 * s1) + (s2 << s3). Assuming s3 is
 * constant, this is an imad instruction. If s1 == 1, then this is optimized to
 * an iadd instruction, which is faster.
 */
static agx_instr *
agx_emit_imadshl_agx(agx_builder *b, nir_alu_instr *alu, agx_index dst,
                     agx_index s0, agx_index s1, agx_index s2, agx_index s3)
{
   /* The immediate iadd/imad encoding accepts only shifts 0..4.  NIR normally
    * guarantees that domain, but this backend is also a process boundary for
    * arbitrary DXIL- and SPIR-V-derived shaders.  Keep every out-of-domain
    * value on the general shift path instead of letting the packer's
    * always-live domain check take down a release process.
    */
   unsigned shift;
   bool constant_shift = nir_src_is_const(alu->src[3].src);

   if (constant_shift)
      shift = nir_alu_src_as_uint(alu->src[3]);

   if (!constant_shift || shift > 4) {
      s2 = agx_bfi(b, agx_immediate(0), s2, s3, 0);
      shift = 0;
   }

   /* Only the immediate-encoding path reaches this assertion. */
   assert(shift <= 4 && "domain restriction on the input NIR");

   /* Emit iadd if possible, else imad */
   if (nir_src_is_const(alu->src[1].src) &&
       nir_alu_src_as_uint(alu->src[1]) == 1) {

      return agx_iadd_to(b, dst, s0, s2, shift);
   } else {
      /*
       * AGX3 has no shifted imad. Apple's own compiler does not emit one:
       * across `o[i] = a[i]*b[i] + (c[i] << N)` kernels for N = 1, 2 and 3,
       * the imadd is BYTE-IDENTICAL and the shift appears on a following
       * iadd instead. So split it the same way rather than inventing a shift
       * field for an instruction that has none.
       *
       * imad(a, b, c, s) is a*b + (c << s), and integer wraparound makes the
       * split exact: the product is computed unshifted, then added with the
       * shift on the addend. Both halves are witnessed byte-exactly: the
       * pure-multiply imadd and the shifted iadd.
       */
      const enum agx_isa_encoding enc = b->shader->key->dev.encoding;
      const bool agx3 = enc != AGX_ISA_ENCODING_AGX2;

      if (agx3 && shift != 0) {
         agx_index product = agx_temp(b->shader, dst.size);
         agx_imad_to(b, product, s0, s1, agx_immediate(0), 0);
         return agx_iadd_to(b, dst, product, s2, shift);
      }

      return agx_imad_to(b, dst, s0, s1, s2, shift);
   }
}

static bool
is_conversion_to_8bit(nir_op op)
{
   switch (op) {
   case nir_op_i2i8:
   case nir_op_u2u8:
   case nir_op_f2i8:
   case nir_op_f2u8:
   case nir_op_b2i8:
      return true;
   default:
      return false;
   }
}

static agx_instr *
agx_fminmax_to(agx_builder *b, agx_index dst, agx_index s0, agx_index s1,
               nir_alu_instr *alu)
{
   /* The hardware gtn/ltn modes are unfortunately incorrect for signed zeros */
   assert(!nir_alu_instr_is_signed_zero_preserve(alu) &&
          "should've been lowered");

   assert((alu->def.bit_size == 16) ==
             (alu->op == nir_op_fmin || alu->op == nir_op_fmax) &&
          "fp32 should be lowered");

   bool fmax = alu->op == nir_op_fmax || alu->op == nir_op_fmax_agx;
   enum agx_fcond fcond = fmax ? AGX_FCOND_GTN : AGX_FCOND_LTN;

   /* Calculate min/max with the appropriate hardware instruction. This will not
    * handle denorms, but we were already lowered for that. On AGX3 the tied
    * fcmpsel is the native fsel condition 3/7 min/max (IEEE minNum/maxNum,
    * the semantics of NIR's fmin/fmax and fmin_agx/fmax_agx); the optimizer
    * keeps both sides tied and the packer never rewrites LTN/GTN into an
    * ordered GE/LE select, which would return the NaN when s0 is NaN.
    */
   return agx_fcmpsel_to(b, dst, s0, s1, s0, s1, fcond);
}

static void
agx_lower_agx3_half_converts(agx_context *ctx)
{
   const bool is_g18g = agx_is_g18g(&ctx->key->dev);
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_CONVERT || I->nr_dests != 1 ||
          I->nr_srcs != 2 ||
          I->src[0].type != AGX_INDEX_IMMEDIATE || I->saturate)
         continue;

      /* G18 half-float to int32 casts also need an exact widening before
       * the native word conversion. Preserve the integer rounding mode. */
      const bool g18_float_to_int =
         ctx->key->dev.emit_agx3d && I->dest[0].size == AGX_SIZE_32 &&
         I->src[1].size == AGX_SIZE_16 && I->round == AGX_ROUND_RTZ &&
         (I->src[0].value == AGX_CONVERT_F_TO_S32 ||
          I->src[0].value == AGX_CONVERT_F_TO_U32);

      /* G18 supports the native signed/unsigned halfword-to-F32 forms.
       * Extend byte inputs into those forms without enabling unrelated
       * AGX3B conversion lowerings. */
      if (is_g18g && !g18_float_to_int &&
          (I->dest[0].size != AGX_SIZE_32 ||
           I->src[1].size != AGX_SIZE_16 || I->round != AGX_ROUND_RTE ||
           (I->src[0].value != AGX_CONVERT_U8_TO_F &&
            I->src[0].value != AGX_CONVERT_S8_TO_F)))
         continue;

      /* M5 converts a signed byte to a half or a float natively (every
       * byte value at every position of a half, both result widths, tested
       * on G17C). G18 keeps the halfword fallback below. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C && !is_g18g &&
          I->src[0].value == AGX_CONVERT_S8_TO_F && I->round == AGX_ROUND_RTE &&
          (I->dest[0].size == AGX_SIZE_16 || I->dest[0].size == AGX_SIZE_32) &&
          I->src[1].size == AGX_SIZE_16 && !I->src[1].abs && !I->src[1].neg &&
          !I->src[1].memory)
         continue;

      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C && !is_g18g &&
          I->src[0].value == AGX_CONVERT_U8_TO_F && I->round == AGX_ROUND_RTE &&
          I->dest[0].size == AGX_SIZE_32 && I->src[1].size == AGX_SIZE_16 &&
          !I->src[1].abs && !I->src[1].neg && !I->src[1].memory)
         continue;

      if (I->src[0].value == AGX_CONVERT_U8_TO_F &&
          I->round == AGX_ROUND_RTE) {
         /* Byte integers occupy a half register; discard the other byte
          * before using the native unsigned-half conversion. */
         agx_builder before = agx_init_builder(ctx, agx_before_instr(I));
         agx_index byte = agx_temp(ctx, AGX_SIZE_16);
         agx_and_to(&before, byte, I->src[1], agx_immediate(0xff));
         I->src[1] = byte;
         I->src[0].value = AGX_CONVERT_U16_TO_F;
      }
      if (I->src[0].value == AGX_CONVERT_S8_TO_F &&
          I->round == AGX_ROUND_RTE) {
         /* Sign-extend the byte inside its half register with the grounded
          * half bit operations and add (x ^ 0x80) - 0x80, then use the
          * native signed-half conversion
          * (spirv_assembly.instruction.*.convertstof.int8_to_float*). */
         agx_builder before = agx_init_builder(ctx, agx_before_instr(I));
         agx_index byte = agx_temp(ctx, AGX_SIZE_16);
         agx_and_to(&before, byte, I->src[1], agx_immediate(0xff));
         agx_index flipped = agx_temp(ctx, AGX_SIZE_16);
         agx_bitop_to(&before, flipped, byte, agx_mov_imm(&before, 16, 0x80),
                      0x6);
         agx_index extended = agx_temp(ctx, AGX_SIZE_16);
         agx_iadd_to(&before, extended, flipped,
                     agx_mov_imm(&before, 16, 0xff80), 0);
         I->src[1] = extended;
         I->src[0].value = AGX_CONVERT_S16_TO_F;
      }
      if (I->dest[0].size == AGX_SIZE_32 &&
          (I->src[0].value == AGX_CONVERT_F_TO_S32 ||
           I->src[0].value == AGX_CONVERT_F_TO_U32) &&
          I->round == AGX_ROUND_RTZ &&
          (I->src[1].size == AGX_SIZE_16 || I->src[1].abs || I->src[1].neg)) {
         /* Every binary16 value is exactly representable in binary32.
          * Widen first and preserve the original integer rounding mode. */
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide, I->src[1]);
         I->src[1] = wide;
         continue;
      }
      /* G18G takes only the conversions above. */
      if (is_g18g)
         continue;
      if (I->dest[0].size != AGX_SIZE_16)
         continue;

      if ((I->src[0].value == AGX_CONVERT_F_TO_S16 ||
           I->src[0].value == AGX_CONVERT_F_TO_U16) &&
          I->round == AGX_ROUND_RTZ) {
         /* NIR f2i16/f2u16 is defined for representable integer results.
          * Convert to the corresponding 32-bit integer then retain its low
          * half. This is exact throughout that defined range, including
          * negative signed results; do not use a floating-point narrowing. */
         agx_builder before = agx_init_builder(ctx, agx_before_instr(I));
         if (I->src[1].size == AGX_SIZE_16 || I->src[1].abs || I->src[1].neg) {
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&before, wide, I->src[1]);
            I->src[1] = wide;
         }
         I->src[0].value = I->src[0].value == AGX_CONVERT_F_TO_S16 ?
            AGX_CONVERT_F_TO_S32 : AGX_CONVERT_F_TO_U32;
         agx_index half = I->dest[0];
         I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
         agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
         agx_subdivide_to(&after, half, I->dest[0], 0);
         continue;
      }

      if (I->round != AGX_ROUND_RTE)
         continue;
      switch (I->src[0].value) {
      case AGX_CONVERT_S32_TO_F:
      case AGX_CONVERT_U32_TO_F:
      case AGX_CONVERT_S16_TO_F:
      case AGX_CONVERT_U16_TO_F:
         break;
      default:
         continue;
      }

      /* Native integer->F32 and F32->F16 are grounded independently. This
       * composition cannot double-round an integer in the finite F16 range:
       * every such integer (including the overflow midpoint) is exact in F32.
       * Larger magnitudes overflow F16 in either route. Run after copyprop so
       * its destination narrowing cannot recreate the unsupported cast.
       */
      agx_index half = I->dest[0];
      I->dest[0] = agx_temp(ctx, AGX_SIZE_32);
      agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
      agx_fmov_to(&b, half, I->dest[0]);
   }
}

/* A modified comparand and the plain value materializing it */
struct agx_m5_comparand {
   agx_index source, plain;
};

static agx_index
agx_m5_find_comparand(const struct util_dynarray *comparands, agx_index src)
{
   util_dynarray_foreach(comparands, struct agx_m5_comparand, c) {
      if (agx_is_equiv(c->source, src) && c->source.size == src.size &&
          c->source.abs == src.abs && c->source.neg == src.neg)
         return c->plain;
   }

   return agx_null();
}

/* A fragment shader that reads a varying array at a run-time index computes
 * its coefficient index as dest16 = imm + (index << 2), because the ITERPROJ
 * coefficient operand is sixteen bits wide. AGX3B's shifted add writes a
 * 32-bit register, so widen the destination and narrow the exact low half. */
/* Form a full-width product from narrow indices with two shifted adds.
 * Keep the extension modifiers on their original operands. */
static void
agx_lower_m5_alu_operands(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C)
      return;

   /* Comparands materialized in the current block. Shaders often compare one
    * |x| against several bounds, and no CSE runs after this pass for M5
    * fragment shaders, so reuse the first move rather than repeat it.
    */
   struct util_dynarray comparands = UTIL_DYNARRAY_INIT;
   agx_block *comparand_block = NULL;

   agx_foreach_instr_global_safe(ctx, I) {
      /* M5 has no grounded compare-select whose compared operands carry |x|
       * or -x: the AGX3C packer refuses every such FCMPSEL form, and FCMP and
       * BREAK_IF_FCMP become FCMPSEL after register allocation. Materialize a
       * modified register/uniform comparand as its own exact move here, after
       * the optimizer has folded what it can, as the agx3c series did. */
      if (I->op == AGX_OPCODE_FCMP || I->op == AGX_OPCODE_FCMPSEL ||
          I->op == AGX_OPCODE_BREAK_IF_FCMP) {
         if (comparand_block != v_block) {
            util_dynarray_clear(&comparands);
            comparand_block = v_block;
         }

         for (unsigned s = 0; s < 2 && s < I->nr_srcs; ++s) {
            const agx_index src = I->src[s];
            if (src.type == AGX_INDEX_IMMEDIATE || !(src.abs || src.neg))
               continue;
            agx_index plain = agx_m5_find_comparand(&comparands, src);
            if (agx_is_null(plain)) {
               agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
               plain = agx_temp(ctx, src.size);
               agx_fmov_to(&b, plain, src);
               util_dynarray_append(
                  &comparands,
                  ((struct agx_m5_comparand){.source = src, .plain = plain}));
            }
            I->src[s] = plain;
         }
      }

      /* A zero-extended low half contributes no bits to a funnel shift
       * by 16 or more. Keep the remaining high-word shift in native form. */
      if (I->op == AGX_OPCODE_EXTR && I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->bfi_mask == 0 && !I->saturate &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[0].size == AGX_SIZE_16 &&
          !I->src[0].abs && !I->src[0].neg &&
          I->src[1].size == AGX_SIZE_32 &&
          !I->src[1].abs && !I->src[1].neg &&
          I->src[2].type == AGX_INDEX_IMMEDIATE &&
          !I->src[2].abs && !I->src[2].neg &&
          I->src[2].value >= 16 && I->src[2].value <= 31) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_bfi_to(&b, I->dest[0], agx_zero(), I->src[1],
                    agx_immediate(32 - I->src[2].value), 0);
         agx_remove_instruction(I);
         continue;
      }

      /* Put a multiplicand's sign into an exact minifloat multiplier. This
       * retains one FMA and avoids materializing a separate negation. */
      if (I->op == AGX_OPCODE_FFMA && I->nr_srcs == 3) {
         const unsigned immediate = I->src[0].type == AGX_INDEX_IMMEDIATE ? 0 : 1;
         agx_index *reg = &I->src[1 - immediate];
         agx_index *imm = &I->src[immediate];
         if (imm->type == AGX_INDEX_IMMEDIATE && imm->value <= UINT8_MAX &&
             !imm->abs && !imm->neg && reg->neg &&
             reg->type != AGX_INDEX_IMMEDIATE) {
            imm->value ^= 0x80;
            reg->neg = false;
         }
      }

      if (I->op == AGX_OPCODE_IADD && I->shift && !I->saturate &&
          I->dest[0].size == AGX_SIZE_32 && I->src[1].size == AGX_SIZE_32 &&
          I->src[1].neg && !I->src[1].abs) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index source = I->src[1];
         source.neg = false;
         agx_index shifted = agx_temp(ctx, AGX_SIZE_32);
         agx_bfi_to(&b, shifted, agx_zero(), source, agx_immediate(I->shift), 0);
         I->src[1] = agx_neg(shifted);
         I->shift = 0;
         continue;
      }
      if (I->op == AGX_OPCODE_FFS && I->src[0].size == AGX_SIZE_16) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[0] = agx_pad_to_32(&b, I->src[0]);
         continue;
      }

      /* A 64-bit product multiplies a word register by a word register, a
       * half register or a small constant. When neither factor is a word
       * register (two halves, a half and a constant), extend one to a word:
       * a plain half is signed and abs selects zero extension, and the word
       * keeps that modifier for its own extension to the product's width. */
      if (I->op == AGX_OPCODE_IMAD && I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_64 && !agx_is_g18g(&ctx->key->dev)) {
         const bool immediate0 = I->src[0].type == AGX_INDEX_IMMEDIATE;
         const bool immediate1 = I->src[1].type == AGX_INDEX_IMMEDIATE;
         if ((!immediate0 && I->src[0].size == AGX_SIZE_32) ||
             (!immediate1 && I->src[1].size == AGX_SIZE_32))
            continue;

         const unsigned s = immediate0 && !immediate1 ? 1 : 0;
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index src = I->src[s];
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         if (src.type == AGX_INDEX_IMMEDIATE) {
            agx_mov_imm_to(&b, wide, src.value);
         } else {
            wide.abs = src.abs;
            src.abs = false;
            if (wide.abs)
               agx_mov_to(&b, wide, src);
            else
               agx_signext_to(&b, wide, src);
         }
         I->src[s] = wide;
         continue;
      }

      if (I->op != AGX_OPCODE_IMAD || I->nr_dests != 1 || I->nr_srcs != 3 ||
          I->saturate || I->shift || I->dest[0].size != AGX_SIZE_32)
         continue;
      /* Metal537 lowers a*b-c to a product and native ISUB. M5's IMADD
       * has no subtract-addend form, so preserve the exact modulo result
       * with those two native operations before register allocation. */
      if (I->src[2].neg) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index addend = I->src[2];
         agx_index product = agx_temp(ctx, I->dest[0].size);
         I->src[2] = agx_zero();
         agx_index dest = I->dest[0];
         I->dest[0] = product;
         b.cursor = agx_after_instr(I);
         agx_iadd_to(&b, dest, product, addend, 0);
         continue;
      }
      /* Native mixed-width multiply-add takes its addend from a register.
       * Materialize a nonzero literal before register allocation. */
      if (I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value &&
          (I->src[0].size == AGX_SIZE_16 || I->src[1].size == AGX_SIZE_16)) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index addend = agx_temp(ctx, AGX_SIZE_32);
         agx_mov_imm_to(&b, addend, I->src[2].value);
         addend.neg = I->src[2].neg;
         I->src[2] = addend;
      }
      const bool swap = I->src[0].type == AGX_INDEX_IMMEDIATE;
      const agx_index a = I->src[swap ? 1 : 0], m = I->src[swap ? 0 : 1];
      if (a.size != AGX_SIZE_16 || a.neg || I->src[2].neg ||
          m.type != AGX_INDEX_IMMEDIATE || m.abs || m.neg ||
          (m.value != 3 && m.value != 6))
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index product = agx_temp(ctx, AGX_SIZE_32);
      agx_iadd_to(&b, product, a, a, 1);
      agx_iadd_to(&b, I->dest[0], I->src[2], product, m.value == 6 ? 1 : 0);
      agx_remove_instruction(I);
   }

   util_dynarray_fini(&comparands);
}

static void
agx_lower_agx3_shifted_iadd_dest(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_IADD || I->nr_dests != 1 || I->nr_srcs != 2 ||
          I->shift == 0 || I->saturate ||
          I->dest[0].size != AGX_SIZE_16 || agx_channels(I->dest[0]) != 1)
         continue;

      /* Only the constant-addend form (agx3_iadd_core.h) is affected: it is
       * the one that always writes a 32-bit register. A shifted add of two
       * registers has its own encoding and keeps its narrow destination. */
      if (I->src[0].type != AGX_INDEX_IMMEDIATE ||
          I->src[1].type == AGX_INDEX_IMMEDIATE)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
      agx_index narrow = I->dest[0];
      agx_index wide = agx_temp(ctx, AGX_SIZE_32);
      I->dest[0] = wide;
      agx_subdivide_to(&b, narrow, wide, 0);
   }
}

/* AGX3B's transcendental forms take one width: agx3_alu_hexp2_neg() for a
 * half pair, agx3_alu_f32exp2_mod() for a word pair, and no mixed form. The
 * exponent of the level-of-detail calculation textureGrad needs arrives as an
 * exp2 of a 16-bit value into a 32-bit result, which was refused -- and a
 * refusal takes the whole shader, which is what
 * KHR-GL33.texture_swizzle's eight remaining smoke cases hit.
 *
 * Widen the source. Half to float is exact, so no value changes; it is the
 * same move agx_lower_agx3_fma_sources() makes for a mixed FFMA. ROUNDEVEN
 * also has separate half and word forms. Widening its half input exactly
 * before rounding preserves the rounded value and signed zero, while using
 * the existing word encoder for the word result. Only the
 * mismatched shape is touched, and that shape did not encode at all before.
 */
void
agx_lower_agx3_mixed_width_unary(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      /* G18G (M6) compiles with the G17G key. Admit only its unsupported
       * scalar half floor; G18G takes none of the other unary rewrites. */
      if (agx_is_g18g(&ctx->key->dev) &&
          !(I->op == AGX_OPCODE_FLOOR && I->nr_dests == 1 &&
            I->dest[0].size == AGX_SIZE_16))
         continue;

      /* The rounding and gradient funops have the same single 32-bit form
       * (agx3_emit_funop()); the optimizer folds an f16 widening into their
       * source exactly as it does for exp2/log2. */
      const bool gradient =
         I->op == AGX_OPCODE_DFDX || I->op == AGX_OPCODE_DFDY;
      if ((I->op != AGX_OPCODE_EXP2 && I->op != AGX_OPCODE_LOG2 &&
           I->op != AGX_OPCODE_TRUNC && I->op != AGX_OPCODE_CEIL &&
           I->op != AGX_OPCODE_FLOOR && I->op != AGX_OPCODE_ROUNDEVEN &&
           !gradient) ||
          I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate)
         continue;

      if (I->src[0].size != AGX_SIZE_16 || agx_channels(I->src[0]) != 1 ||
          agx_channels(I->dest[0]) != 1 ||
          I->src[0].type == AGX_INDEX_IMMEDIATE ||
          I->src[0].type == AGX_INDEX_NULL)
         continue;

      /* A half gradient has no AGX3 form at all. The difference of two
       * halves is exact in f32, so differentiating the widened value and
       * rounding once gives the correctly rounded half result. */
      const bool half_gradient = gradient && I->dest[0].size == AGX_SIZE_16;
      /* G18 sample-variable shaders also reach a scalar half floor without
       * a grounded native form. Every finite half and its floor are exactly
       * representable in f32, so widen, round, and narrow after optimization.
       * Keep the conversion explicit through register allocation. */
      const bool half_floor = ctx->key->dev.emit_agx3d &&
                              I->op == AGX_OPCODE_FLOOR &&
                              I->dest[0].size == AGX_SIZE_16;
      if (I->dest[0].size != AGX_SIZE_32 && !half_gradient && !half_floor)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index widened = agx_temp(ctx, AGX_SIZE_32);
      if (I->op == AGX_OPCODE_ROUNDEVEN) {
         /* Preserve source signs on the exact half-to-word conversion. */
         agx_fmov_to(&b, widened, I->src[0]);
      } else {
         /* Retain the base's direct funop modifiers for the other forms. */
         agx_index plain = I->src[0];
         plain.abs = plain.neg = false;
         agx_fmov_to(&b, widened, plain);
         widened.abs = I->src[0].abs;
         widened.neg = I->src[0].neg;
      }
      I->src[0] = widened;

      if (half_gradient || half_floor) {
         agx_index narrow = I->dest[0];
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         I->dest[0] = wide;
         b.cursor = agx_after_instr(I);
         agx_fmov_to(&b, narrow, wide);
      }
   }
}

/* M5 legalization widens a half operand with a float move, fadd(x, -0.0),
 * where an f32 operation cannot read it. agx_opt_compact_constants made
 * such an operand a half constant (1/256 in Minecraft's terrain vertex
 * shader), so the vertex shader materialized the half and then moved it to
 * f32 twice for every vertex. A float move of a known finite, normal
 * constant is that constant exactly (f16 -> f32 is exact; x + -0.0 == x
 * for every non-NaN x, including +-0), so materialize the f32 bits once
 * instead. NaNs, infinities and f32 denormals keep their moves. */
static void
agx3_fold_constant_fmovs(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C)
      return;

   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
   if (!defs)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;

      if (I->op != AGX_OPCODE_FADD || I->nr_dests != 1 || I->saturate ||
          I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
          I->src[0].type != AGX_INDEX_NORMAL || I->src[0].abs ||
          I->src[0].neg || I->src[0].memory ||
          !agx_is_equiv(I->src[1], agx_negzero()) || I->src[1].abs ||
          I->src[1].neg)
         continue;

      const agx_instr *def = defs[I->src[0].value];
      if (!def || def->op != AGX_OPCODE_MOV_IMM || def->nr_dests != 1 ||
          agx_channels(def->dest[0]) != 1 ||
          (def->dest[0].size != AGX_SIZE_16 &&
           def->dest[0].size != AGX_SIZE_32))
         continue;

      const float value = def->dest[0].size == AGX_SIZE_16
                             ? _mesa_half_to_float((uint16_t)def->imm)
                             : uif((uint32_t)def->imm);
      const int cls = fpclassify(value);
      if (cls != FP_NORMAL && cls != FP_ZERO)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_instr *mov = agx_mov_imm_to(&b, I->dest[0], fui(value));
      defs[I->dest[0].value] = mov;
      agx_remove_instruction(I);
   }

   free(defs);
}

void
agx_lower_agx3_fma_sources(agx_context *ctx)
{
   /* G18G keeps j873's fused G17G-family half widening (15f6b6a7bd2). */
   const bool g18g = agx_is_g18g(&ctx->key->dev);
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_FFMA || I->nr_srcs != 3 || I->saturate)
         continue;

      if (g18g && I->dest[0].size != AGX_SIZE_32)
         continue;
      /* Both literal minifloats are finite. If either is signed zero, their
       * exact product is zero with the XOR of their signs. Keep the floating
       * point addition with that signed zero: a bitwise move would change
       * signed-zero, denormal or NaN behavior, and replacing -0 with +0 would
       * change the result for a -0 addend. No run-time multiplier is folded.
       */
      if (I->nr_dests == 1 && I->dest[0].size == AGX_SIZE_32 &&
          agx_channels(I->dest[0]) == 1 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE && I->src[0].value <= 0xff &&
          !I->src[0].abs && !I->src[0].neg && !I->src[0].memory &&
          I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value <= 0xff &&
          !I->src[1].abs && !I->src[1].neg && !I->src[1].memory &&
          ((I->src[0].value & 0x7f) == 0 ||
           (I->src[1].value & 0x7f) == 0) &&
          I->src[2].type != AGX_INDEX_NULL &&
          I->src[2].size == AGX_SIZE_32 && agx_channels(I->src[2]) == 1) {
         const unsigned sign = (I->src[0].value ^ I->src[1].value) & 0x80;
         I->op = AGX_OPCODE_FADD;
         I->nr_srcs = 2;
         I->src[0] = I->src[2];
         I->src[1] = agx_immediate(sign);
         continue;
      }

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      bool has_full_gpr = false;
      unsigned half_uniforms = 0;
      agx_foreach_src(I, s) {
         has_full_gpr |= (I->src[s].type == AGX_INDEX_NORMAL ||
                          I->src[s].type == AGX_INDEX_REGISTER) &&
                         I->src[s].size == AGX_SIZE_32;
         half_uniforms += I->src[s].type == AGX_INDEX_UNIFORM &&
                          I->src[s].size == AGX_SIZE_16;
      }
      agx_foreach_src(I, s) {
         if (I->src[s].size != AGX_SIZE_16 ||
             I->src[s].type == AGX_INDEX_IMMEDIATE ||
             I->src[s].type == AGX_INDEX_NULL ||
             (!g18g && (I->src[s].abs || I->src[s].neg)))
            continue;
         if (I->nr_dests == 1 && I->dest[0].size == AGX_SIZE_32 &&
             agx_channels(I->dest[0]) == 1 && has_full_gpr &&
             half_uniforms == 1 && I->src[s].type == AGX_INDEX_UNIFORM)
            continue;

         agx_index widened = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, widened, I->src[s]);
         I->src[s] = widened;
      }

      /* G18G's (G17G-family) native half widening is exact, including
       * source modifiers. Widen before the full-register FMA to preserve its
       * single rounding; do not split the fused operation into multiply and
       * add. The modifier workarounds below describe other instruction forms.
       */
      if (g18g)
         continue;

      bool mixed = false, has_modifier = false, unsupported_modifier = false;
      agx_foreach_src(I, s) {
         mixed |= I->src[s].type == AGX_INDEX_IMMEDIATE ||
                  I->src[s].size != AGX_SIZE_32;
         has_modifier |= I->src[s].abs || I->src[s].neg;
         unsupported_modifier |= ((I->src[s].abs || I->src[s].neg) &&
                                  (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                                   I->src[s].size != AGX_SIZE_32));
      }

      /* The native modifier form takes three full F32 registers. Preserve
       * exact fabs/fneg semantics for mixed/minifloat FMA shapes by clearing
       * and then flipping the source sign bit before register allocation. The resulting FMA has no
       * modifier and stays on the already-grounded mixed-source form. */
      if (!mixed || !has_modifier || unsupported_modifier)
         continue;

      /* Two mixed shapes have native modifier-bearing encodings that the
       * packer emits directly, so keep their modifiers fused instead of
       * spending a MOV-immediate and a BITOP per modified source:
       *
       *  - two F32 register multiplicands with abs/neg and a plain minifloat
       *    addend (agx3_falu_fma_addend_imm_mod; Apple G17P emits
       *    fma(-a, b, 1.0f) as one f32fma, 3903060121b002c8);
       *  - one plain minifloat multiplier with a modified register
       *    multiplicand and addend of either width
       *    (agx3_falu_fma_mul_imm_mod).
       *
       * Both still round once, exactly like the materialized sign bits. */
      if (I->nr_dests == 1 && I->dest[0].size == AGX_SIZE_32 &&
          agx_channels(I->dest[0]) == 1) {
         const bool imm0 = I->src[0].type == AGX_INDEX_IMMEDIATE;
         const bool imm1 = I->src[1].type == AGX_INDEX_IMMEDIATE;
         const bool imm2 = I->src[2].type == AGX_INDEX_IMMEDIATE;
         bool native_shape = false;
         if (imm2 && !imm0 && !imm1) {
            native_shape = I->src[2].value <= 0xff &&
                           I->src[0].size == AGX_SIZE_32 &&
                           I->src[1].size == AGX_SIZE_32 &&
                           agx_channels(I->src[0]) == 1 &&
                           agx_channels(I->src[1]) == 1 &&
                           !I->src[0].memory && !I->src[1].memory;
         } else if (imm0 != imm1 && !imm2) {
            const agx_index imm = I->src[imm0 ? 0 : 1];
            const agx_index reg = I->src[imm0 ? 1 : 0];
            native_shape = imm.value <= 0xff && !imm.memory &&
                           agx_channels(reg) == 1 &&
                           agx_channels(I->src[2]) == 1 &&
                           !reg.memory && !I->src[2].memory &&
                           (reg.size == AGX_SIZE_32 ||
                            reg.size == AGX_SIZE_16) &&
                           (I->src[2].size == AGX_SIZE_32 ||
                            I->src[2].size == AGX_SIZE_16);
         }
         if (native_shape)
            continue;
      }

      agx_foreach_src(I, s) {
         const bool absolute = I->src[s].abs, negate = I->src[s].neg;
         if (!absolute && !negate)
            continue;

         agx_index source = I->src[s];
         source.abs = source.neg = false;
         if (absolute) {
            agx_index mask = agx_mov_imm(&b, 32, UINT32_C(0x7fffffff));
            agx_index cleared = agx_temp(ctx, AGX_SIZE_32);
            agx_bitop_to(&b, cleared, source, mask, 0x8);
            source = cleared;
         }
         if (negate) {
            agx_index sign = agx_mov_imm(&b, 32, UINT32_C(0x80000000));
            agx_index flipped = agx_temp(ctx, AGX_SIZE_32);
            agx_bitop_to(&b, flipped, source, sign, 0x6);
            source = flipped;
         }
         I->src[s] = source;
      }
   }
}

static agx_index
agx3_materialize_f32_source(agx_builder *b, agx_index source)
{
   assert(source.type != AGX_INDEX_IMMEDIATE);
   assert(source.size == AGX_SIZE_16 || source.size == AGX_SIZE_32);
   const bool absolute = source.abs, negate = source.neg;
   source.abs = source.neg = false;

   if (source.size == AGX_SIZE_16) {
      agx_index widened = agx_temp(b->shader, AGX_SIZE_32);
      agx_fmov_to(b, widened, source);
      source = widened;
   }

   /* IEEE abs/neg are exact sign-bit operations, including signed zero. */
   if (absolute) {
      agx_index mask = agx_mov_imm(b, 32, UINT32_C(0x7fffffff));
      agx_index result = agx_temp(b->shader, AGX_SIZE_32);
      agx_bitop_to(b, result, source, mask, 0x8 /* AND */);
      source = result;
   }
   if (negate) {
      agx_index sign = agx_mov_imm(b, 32, UINT32_C(0x80000000));
      agx_index result = agx_temp(b->shader, AGX_SIZE_32);
      agx_bitop_to(b, result, source, sign, 0x6 /* XOR */);
      source = result;
   }
   return source;
}

/*
 * Second G17G attempt only (ctx->agx3_reg_float_consts): many float ALU shapes
 * with a minifloat immediate, or with a half compared against an f32, have no
 * grounded G17G encoding, while the plain register forms do. Materialize each
 * float immediate as an exact register of the width the operand is read at
 * (comparands as f32, everything else at the destination width) and widen a
 * half comparand facing an f32 one. Binary16 values and minifloats are exact
 * in both widths, so results are unchanged.
 */
bool
agx_lower_g17g_retry_float_operands(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       !ctx->agx3_reg_float_consts)
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->nr_dests != 1 || I->dest[0].type != AGX_INDEX_NORMAL)
         continue;

      /* G17G has no witnessed unordered LTN/GTN compare-select. They are
       * exactly the negations of the ordered GE/LE (NaN included): swap the
       * selected values, or flip an unfused compare's inversion. */
      if ((I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4) ||
          I->op == AGX_OPCODE_FCMP) {
         if (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) {
            I->fcond = I->fcond == AGX_FCOND_LTN ? AGX_FCOND_GE : AGX_FCOND_LE;
            if (I->op == AGX_OPCODE_FCMPSEL) {
               agx_index t = I->src[2];
               I->src[2] = I->src[3];
               I->src[3] = t;
            } else {
               I->invert_cond = !I->invert_cond;
            }
            progress = true;
         }
      }

      /* An unfused signed integer comparison is lowered after RA to a 0/1
       * select; G17G grounds signed conditions only with a 32-bit result
       * (a loop counter's `i < n` in a 16-bit boolean). Compare into a
       * 32-bit temporary and narrow it. */
      /* With half comparands (mediump `a == b` into a boolean), RA turns the
       * compare into a half 0/1 select on half registers, which G17G has no
       * form for under any condition. Compare 32-bit extensions instead:
       * signed (SEQ too) or zero extended, sext as in SIGNEXT above. */
      if (I->op == AGX_OPCODE_ICMP && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_16 &&
          I->src[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16) {
         bool ok = true;
         for (unsigned s = 0; s < 2; ++s) {
            const agx_index src = I->src[s];
            ok &= !src.neg && !src.abs && agx_channels(src) == 1 &&
                  (src.type == AGX_INDEX_NORMAL ||
                   src.type == AGX_INDEX_UNIFORM ||
                   src.type == AGX_INDEX_IMMEDIATE);
         }
         if (ok) {
            const bool is_signed = I->icond == AGX_ICOND_SLT ||
                                   I->icond == AGX_ICOND_SGT ||
                                   I->icond == AGX_ICOND_SEQ;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            for (unsigned s = 0; s < 2; ++s) {
               const agx_index src = I->src[s];
               if (src.type == AGX_INDEX_IMMEDIATE) {
                  I->src[s] = agx_mov_imm(
                     &b, 32,
                     is_signed ? (uint32_t)(int32_t)(int16_t)src.value
                               : src.value);
                  continue;
               }
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_mov_to(&b, wide, src);
               if (is_signed) {
                  agx_index bias = agx_mov_imm(&b, 32, 0x8000);
                  agx_index flipped = agx_temp(ctx, AGX_SIZE_32);
                  agx_bitop_to(&b, flipped, wide, bias, 0x6 /* XOR */);
                  agx_index extended = agx_temp(ctx, AGX_SIZE_32);
                  agx_iadd_to(&b, extended, flipped, agx_neg(bias), 0);
                  wide = extended;
               }
               I->src[s] = wide;
            }
            agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
            agx_index narrow = I->dest[0];
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            I->dest[0] = wide;
            agx_subdivide_to(&after, narrow, wide, 0);
            progress = true;
            continue;
         }
      }

      if (I->op == AGX_OPCODE_ICMP && I->dest[0].size == AGX_SIZE_16 &&
          (I->icond == AGX_ICOND_SLT || I->icond == AGX_ICOND_SGT)) {
         agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
         agx_index narrow = I->dest[0];
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         I->dest[0] = wide;
         agx_subdivide_to(&b, narrow, wide, 0);
         progress = true;
         continue;
      }

      /* An integer compare-select with a half result has no grounded G17G
       * form for the signed conditions, nor for an immediate comparand or
       * immediate selected values (GTK's Vulkan vertex shaders select 16-bit
       * values on a 32-bit `x < n`). Select between 32-bit registers into a
       * 32-bit temporary and narrow it, as for ICMP above: the selected
       * values are bit patterns and are zero-extended; an immediate
       * comparand takes the width of the register it is compared with. */
      /* With half comparands too (mediump `a < b ? 1 : 0`, registers or
       * uniforms, which are only lowered later), widen them first:
       * sign-extended for the signed conditions (SEQ included, so that an
       * immediate comparand extends the same way as the register), zero-
       * extended otherwise. G17G has no witnessed sign-extending half add,
       * so extend as sext(x) = (zext(x) ^ 0x8000) - 0x8000, as SIGNEXT is
       * lowered above. Every operand is checked before any is rewritten. */
      /* An immediate first comparand: compare the other way round. */
      if (I->op == AGX_OPCODE_ICMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE &&
          (I->src[1].type == AGX_INDEX_NORMAL ||
           I->src[1].type == AGX_INDEX_UNIFORM)) {
         const agx_index t = I->src[0];
         I->src[0] = I->src[1];
         I->src[1] = t;
         switch (I->icond) {
         case AGX_ICOND_ULT: I->icond = AGX_ICOND_UGT; break;
         case AGX_ICOND_UGT: I->icond = AGX_ICOND_ULT; break;
         case AGX_ICOND_SLT: I->icond = AGX_ICOND_SGT; break;
         case AGX_ICOND_SGT: I->icond = AGX_ICOND_SLT; break;
         default: break;
         }
         progress = true;
      }

      if (I->op == AGX_OPCODE_ICMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 &&
          (I->src[0].type == AGX_INDEX_NORMAL ||
           I->src[0].type == AGX_INDEX_UNIFORM) &&
          (I->src[0].size == AGX_SIZE_32 || I->src[0].size == AGX_SIZE_16)) {
         const bool cmp16 = I->src[0].size == AGX_SIZE_16;
         const bool is_signed = I->icond == AGX_ICOND_SLT ||
                                I->icond == AGX_ICOND_SGT ||
                                I->icond == AGX_ICOND_SEQ;
         bool ok = true;
         for (unsigned s = 0; s < 4 && ok; ++s) {
            const agx_index src = I->src[s];
            if (src.neg || src.abs)
               ok = false;
            else if (src.type == AGX_INDEX_IMMEDIATE)
               ok = s != 0;
            else if ((src.type != AGX_INDEX_NORMAL &&
                      src.type != AGX_INDEX_UNIFORM) ||
                     agx_channels(src) != 1)
               ok = false;
            else if (s < 2)
               ok = src.size == I->src[0].size;
            else
               ok = src.size == AGX_SIZE_16 || src.size == AGX_SIZE_32;
         }
         if (!ok)
            continue;

         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         for (unsigned s = 0; s < 4; ++s) {
            agx_index src = I->src[s];
            const bool compared = s < 2;
            if (src.type == AGX_INDEX_IMMEDIATE) {
               uint32_t value = src.value;
               if (compared && cmp16 && is_signed)
                  value = (uint32_t)(int32_t)(int16_t)value;
               I->src[s] = agx_mov_imm(&b, 32, value);
            } else if (src.size == AGX_SIZE_16 &&
                       !(compared && !cmp16)) {
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_mov_to(&b, wide, src);
               if (compared && is_signed) {
                  agx_index bias = agx_mov_imm(&b, 32, 0x8000);
                  agx_index flipped = agx_temp(ctx, AGX_SIZE_32);
                  agx_bitop_to(&b, flipped, wide, bias, 0x6 /* XOR */);
                  agx_index extended = agx_temp(ctx, AGX_SIZE_32);
                  agx_iadd_to(&b, extended, flipped, agx_neg(bias), 0);
                  wide = extended;
               }
               I->src[s] = wide;
            }
         }
         {
            agx_index narrow = I->dest[0];
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            I->dest[0] = wide;
            agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
            agx_subdivide_to(&after, narrow, wide, 0);
            progress = true;
         }
         continue;
      }

      /* A 32-bit result selected on half comparands (a bool `==` in a vertex
       * shader) is grounded on G17G only against an equality immediate of
       * at most 2. Compare the 32-bit extensions instead, sign-extended for
       * the signed conditions and zero-extended otherwise, as above. */
      if (I->op == AGX_OPCODE_ICMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 && I->src[0].size == AGX_SIZE_16 &&
          (I->src[0].type == AGX_INDEX_NORMAL ||
           I->src[0].type == AGX_INDEX_UNIFORM) &&
          !(I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value <= 2 &&
            (I->icond == AGX_ICOND_UEQ || I->icond == AGX_ICOND_SEQ))) {
         const bool is_signed = I->icond == AGX_ICOND_SLT ||
                                I->icond == AGX_ICOND_SGT ||
                                I->icond == AGX_ICOND_SEQ;
         bool ok = true;
         for (unsigned s = 0; s < 2 && ok; ++s) {
            const agx_index src = I->src[s];
            ok = !src.neg && !src.abs &&
                 (src.type == AGX_INDEX_IMMEDIATE ||
                  ((src.type == AGX_INDEX_NORMAL ||
                    src.type == AGX_INDEX_UNIFORM) &&
                   agx_channels(src) == 1 && src.size == AGX_SIZE_16));
         }
         if (ok) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            for (unsigned s = 0; s < 2; ++s) {
               const agx_index src = I->src[s];
               if (src.type == AGX_INDEX_IMMEDIATE) {
                  I->src[s] = agx_mov_imm(
                     &b, 32,
                     is_signed ? (uint32_t)(int32_t)(int16_t)src.value
                               : src.value);
                  continue;
               }
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_mov_to(&b, wide, src);
               if (is_signed) {
                  agx_index bias = agx_mov_imm(&b, 32, 0x8000);
                  agx_index flipped = agx_temp(ctx, AGX_SIZE_32);
                  agx_bitop_to(&b, flipped, wide, bias, 0x6 /* XOR */);
                  agx_index extended = agx_temp(ctx, AGX_SIZE_32);
                  agx_iadd_to(&b, extended, flipped, agx_neg(bias), 0);
                  wide = extended;
               }
               I->src[s] = wide;
            }
            progress = true;
            continue;
         }
      }

      /* G17G grounds a & ~b (truth table 0x2, 0x4 with the sources
       * exchanged) only on halves. At 32 bits complement the cleared source
       * with an XOR against all ones and AND (bitfieldInsert's select). */
      if (I->op == AGX_OPCODE_BITOP && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_32 &&
          ((I->truth_table & 0xfu) == 0x2u ||
           (I->truth_table & 0xfu) == 0x4u)) {
         const unsigned c = (I->truth_table & 0xfu) == 0x2u ? 1 : 0;
         const agx_index src = I->src[c];
         if (!src.neg && !src.abs && src.size == AGX_SIZE_32 &&
             agx_channels(src) == 1 &&
             (src.type == AGX_INDEX_NORMAL || src.type == AGX_INDEX_UNIFORM ||
              src.type == AGX_INDEX_IMMEDIATE)) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            if (src.type == AGX_INDEX_IMMEDIATE) {
               I->src[c] = agx_mov_imm(&b, 32, ~(uint32_t)src.value);
            } else {
               agx_index inverted = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, inverted, src,
                            agx_mov_imm(&b, 32, UINT32_C(0xffffffff)),
                            0x6 /* XOR */);
               I->src[c] = inverted;
            }
            I->truth_table = 0x8 /* AND */;
            progress = true;
            continue;
         }
      }

      /* |a| * |b| at half precision has no grounded G17G form with both
       * sources absolute: multiply and clear the sign bit (|a*b|, NaNs and
       * signed zeros included). */
      if ((I->op == AGX_OPCODE_FMUL || I->op == AGX_OPCODE_HMUL) &&
          I->nr_srcs == 2 && !I->saturate &&
          I->dest[0].size == AGX_SIZE_16 &&
          I->src[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16 &&
          I->src[0].abs && I->src[1].abs &&
          !I->src[0].neg && !I->src[1].neg &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].type != AGX_INDEX_IMMEDIATE) {
         I->src[0].abs = I->src[1].abs = false;
         agx_index product = agx_temp(ctx, AGX_SIZE_16);
         agx_index narrow = I->dest[0];
         I->dest[0] = product;
         agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
         agx_bitop_to(&after, narrow, product,
                      agx_mov_imm(&after, 16, 0x7fff), 0x8 /* AND */);
         progress = true;
         continue;
      }

      /* An f32 -> f16 conversion of a negated or absolute source
       * (fadd(-x, -0.0) into a half) has no grounded form; convert x and
       * set the half's sign: XOR 0x8000 for -x, AND 0x7fff for |x|, OR
       * 0x8000 for -|x|. */
      if (I->op == AGX_OPCODE_FADD && I->nr_srcs == 2 && !I->saturate &&
          I->dest[0].size == AGX_SIZE_16 && I->src[0].size == AGX_SIZE_32 &&
          (I->src[0].neg || I->src[0].abs) &&
          (I->src[0].type == AGX_INDEX_NORMAL ||
           I->src[0].type == AGX_INDEX_UNIFORM) &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].value == agx_negzero().value &&
          I->src[1].neg == agx_negzero().neg &&
          I->src[1].abs == agx_negzero().abs) {
         const bool abs = I->src[0].abs, neg = I->src[0].neg;
         I->src[0].neg = I->src[0].abs = false;
         agx_index converted = agx_temp(ctx, AGX_SIZE_16);
         agx_index narrow = I->dest[0];
         I->dest[0] = converted;
         agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
         agx_bitop_to(&after, narrow, converted,
                      agx_mov_imm(&after, 16, abs && !neg ? 0x7fff : 0x8000),
                      abs ? (neg ? 0xe /* OR */ : 0x8 /* AND */)
                          : 0x6 /* XOR */);
         progress = true;
         continue;
      }

      /* G17G grounds only the 32-bit integer -> float conversions (to f32
       * and to f16); the byte and half forms are G17P's. Extend the source
       * to 32 bits first: zero extension, masked to the byte for U8; sign
       * extension as (x ^ m) - m with m = 0x80 or 0x8000. */
      /* A byte is exact in f16 and f32, a half integer in f32, so the
       * rounding mode of those conversions does not matter. */
      if (I->op == AGX_OPCODE_CONVERT && I->nr_srcs == 2 && !I->saturate &&
          (I->round == AGX_ROUND_RTE || I->dest[0].size == AGX_SIZE_32 ||
           (I->src[0].type == AGX_INDEX_IMMEDIATE &&
            (I->src[0].value == AGX_CONVERT_U8_TO_F ||
             I->src[0].value == AGX_CONVERT_S8_TO_F))) &&
          I->src[0].type == AGX_INDEX_IMMEDIATE &&
          (I->src[0].value == AGX_CONVERT_U8_TO_F ||
           I->src[0].value == AGX_CONVERT_S8_TO_F ||
           I->src[0].value == AGX_CONVERT_U16_TO_F ||
           I->src[0].value == AGX_CONVERT_S16_TO_F) &&
          I->src[1].size == AGX_SIZE_16 &&
          (I->src[1].type == AGX_INDEX_NORMAL ||
           I->src[1].type == AGX_INDEX_UNIFORM) &&
          agx_channels(I->src[1]) == 1 && !I->src[1].abs && !I->src[1].neg) {
         const enum agx_convert mode = I->src[0].value;
         const bool byte =
            mode == AGX_CONVERT_U8_TO_F || mode == AGX_CONVERT_S8_TO_F;
         const bool is_signed =
            mode == AGX_CONVERT_S8_TO_F || mode == AGX_CONVERT_S16_TO_F;
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_mov_to(&b, wide, I->src[1]);
         if (byte) {
            agx_index masked = agx_temp(ctx, AGX_SIZE_32);
            agx_bitop_to(&b, masked, wide, agx_mov_imm(&b, 32, 0xff),
                         0x8 /* AND */);
            wide = masked;
         }
         if (is_signed) {
            agx_index bias = agx_mov_imm(&b, 32, byte ? 0x80 : 0x8000);
            agx_index flipped = agx_temp(ctx, AGX_SIZE_32);
            agx_bitop_to(&b, flipped, wide, bias, 0x6 /* XOR */);
            agx_index extended = agx_temp(ctx, AGX_SIZE_32);
            agx_iadd_to(&b, extended, flipped, agx_neg(bias), 0);
            wide = extended;
         }
         I->src[0] = agx_immediate(is_signed ? AGX_CONVERT_S32_TO_F
                                             : AGX_CONVERT_U32_TO_F);
         I->src[1] = wide;
         I->round = AGX_ROUND_RTE;
         progress = true;
         continue;
      }

      /* A three-channel tile load (blending into RGB565) has no grounded
       * G17G form: load all four channels and keep the first three. */
      if (I->op == AGX_OPCODE_LD_TILE && I->mask == 0x7 &&
          agx_channels(I->dest[0]) == 3) {
         agx_index narrow = I->dest[0];
         agx_index wide = agx_vec_temp(ctx, narrow.size, 4);
         I->dest[0] = wide;
         I->mask = 0xf;
         agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
         agx_index channels[4];
         agx_emit_split(&after, channels, wide, 4);
         agx_emit_collect_to(&after, narrow, 3, channels);
         progress = true;
         continue;
      }

      /* Half derivatives have no grounded G17G form: differentiate the f32
       * extension and round the result back (f16 -> f32 is exact, and the
       * f32 difference rounds once to the same half the half unit would
       * produce for representable values). */
      if ((I->op == AGX_OPCODE_DFDX || I->op == AGX_OPCODE_DFDY) &&
          I->nr_srcs == 1 && I->dest[0].size == AGX_SIZE_16 &&
          I->src[0].size == AGX_SIZE_16 && !I->saturate &&
          (I->src[0].type == AGX_INDEX_NORMAL ||
           I->src[0].type == AGX_INDEX_UNIFORM) &&
          agx_channels(I->src[0]) == 1) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide_src = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide_src, I->src[0]);
         agx_index wide_dst = agx_temp(ctx, AGX_SIZE_32);
         agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
         agx_fmov_to(&after, I->dest[0], wide_dst);
         I->src[0] = wide_src;
         I->dest[0] = wide_dst;
         progress = true;
         continue;
      }

      /* `imm - x` at 32 bits has no G17G form with the immediate in the
       * first slot; register subtraction is witnessed. The immediate keeps
       * its extension: .abs marks a zero-extended source, as for registers
       * in agx_lower_agx3_widening_iadd(). */
      if (I->op == AGX_OPCODE_IADD && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE && !I->src[0].neg &&
          (I->src[1].type == AGX_INDEX_NORMAL ||
           I->src[1].type == AGX_INDEX_UNIFORM) &&
          I->src[1].neg && I->src[1].size == AGX_SIZE_32) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         const agx_index imm = I->src[0];
         I->src[0] = agx_mov_imm(
            &b, 32, imm.abs ? imm.value : (uint32_t)(int32_t)(int16_t)imm.value);
         progress = true;
         continue;
      }

      /* An unfused float comparison (lowered after RA to a 0/1 select) has
       * no mixed half/f32 form either: widen the half comparand. */
      if (I->op == AGX_OPCODE_FCMP && I->nr_srcs == 2) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type != AGX_INDEX_NORMAL ||
                I->src[s].size != AGX_SIZE_16)
               continue;
            agx_index plain = I->src[s];
            plain.abs = false;
            plain.neg = false;
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, wide, plain);
            wide.abs = I->src[s].abs;
            wide.neg = I->src[s].neg;
            I->src[s] = wide;
            progress = true;
         }

         /* The post-RA lowering selects immediates 1 and 0, and G17G has no
          * grounded register-compare/immediate-select form at either result
          * width (WebRender's box-shadow fragment shader). Select between
          * registers holding 1 and 0 here instead, at 32 bits, and narrow
          * an integer half result afterwards. */
         if (I->src[0].type == AGX_INDEX_NORMAL &&
             I->src[1].type == AGX_INDEX_NORMAL &&
             I->src[0].size == AGX_SIZE_32 && I->src[1].size == AGX_SIZE_32 &&
             I->nr_dests == 1 &&
             (I->dest[0].size == AGX_SIZE_16 ||
              I->dest[0].size == AGX_SIZE_32)) {
            agx_index one = agx_mov_imm(&b, 32, 1);
            agx_index zero = agx_mov_imm(&b, 32, 0);
            agx_index wide = I->dest[0].size == AGX_SIZE_32
                                ? I->dest[0]
                                : agx_temp(ctx, AGX_SIZE_32);
            agx_fcmpsel_to(&b, wide, I->src[0], I->src[1],
                           I->invert_cond ? zero : one,
                           I->invert_cond ? one : zero, I->fcond);
            if (I->dest[0].size == AGX_SIZE_16)
               agx_subdivide_to(&b, I->dest[0], wide, 0);
            agx_remove_instruction(I);
            progress = true;
         }
         continue;
      }

      if (I->op != AGX_OPCODE_FCMPSEL && I->op != AGX_OPCODE_FFMA &&
          I->op != AGX_OPCODE_FADD && I->op != AGX_OPCODE_FMUL)
         continue;

      /* A half fabs/fneg, fadd(h.abs/.neg, -0.0) into a half, has no
       * grounded G17G form. It only touches the sign bit: AND 0x7fff for
       * |h|, XOR 0x8000 for -h, OR 0x8000 for -|h|, with the 16-bit bit
       * operations G17G does encode (see the sign extension above). */
      if (I->op == AGX_OPCODE_FADD && I->nr_srcs == 2 && !I->saturate &&
          I->dest[0].size == AGX_SIZE_16) {
         const agx_index z = agx_negzero();
         int x_slot = -1;
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE &&
                I->src[s].value == z.value && I->src[s].neg == z.neg &&
                I->src[s].abs == z.abs)
               x_slot = !s;
         }
         agx_index x = x_slot >= 0 ? I->src[x_slot] : agx_null();
         if (x_slot >= 0 &&
             (x.type == AGX_INDEX_NORMAL || x.type == AGX_INDEX_UNIFORM) &&
             x.size == AGX_SIZE_16 && agx_channels(x) == 1 &&
             (x.abs || x.neg)) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            const bool abs = x.abs, neg = x.neg;
            x.abs = x.neg = false;
            agx_index mask =
               agx_mov_imm(&b, 16, abs && !neg ? 0x7fff : 0x8000);
            agx_bitop_to(&b, I->dest[0], x, mask,
                         abs ? (neg ? 0xe /* OR */ : 0x8 /* AND */)
                             : 0x6 /* XOR */);
            agx_remove_instruction(I);
            progress = true;
            continue;
         }
      }

      /* fadd(x, -0.0) is agx_fmov_to(), i.e. a conversion or plain move
       * with its own grounded forms: never pull its -0.0 into a register. */
      if (I->op == AGX_OPCODE_FADD && I->nr_srcs == 2) {
         const agx_index z = agx_negzero();
         bool is_fmov = false;
         for (unsigned s = 0; s < 2; ++s)
            is_fmov |= I->src[s].type == AGX_INDEX_IMMEDIATE &&
                       I->src[s].value == z.value &&
                       I->src[s].neg == z.neg && I->src[s].abs == z.abs;
         if (is_fmov)
            continue;
      }

      const bool select = I->op == AGX_OPCODE_FCMPSEL;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         agx_index src = I->src[s];
         if (src.type != AGX_INDEX_IMMEDIATE)
            continue;

         const bool comparand = select && s < 2;
         enum agx_size size = comparand ? AGX_SIZE_32 : I->dest[0].size;

         /* The selected values of a compare-select are raw bit patterns,
          * not floats (agx_is_float_src): keep them bit-exact. Their ABS
          * flag only marks a 16-bit immediate. */
         if (!agx_is_float_src(I, s)) {
            if (src.neg || (size != AGX_SIZE_16 && size != AGX_SIZE_32))
               continue;
            I->src[s] = agx_mov_imm(&b, size == AGX_SIZE_32 ? 32 : 16,
                                    src.value);
            progress = true;
            continue;
         }

         float value = agx_minifloat_decode(src.value);
         if (src.abs)
            value = fabsf(value);
         if (src.neg)
            value = -value;

         if (size == AGX_SIZE_32) {
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            I->src[s] = agx_mov_imm(&b, 32, bits);
         } else if (size == AGX_SIZE_16) {
            I->src[s] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         } else {
            continue;
         }
         progress = true;
      }

      /* Mixed half/f32 operands have no G17G form for these operations:
       * compute at f32. A half destination fed by f32 operands gets an f32
       * temporary and one conversion afterwards (exact for a select, and a
       * single rounding for arithmetic, as the native mixed op would do). */
      bool any32 = false, any16 = false;
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         if (I->src[s].type != AGX_INDEX_NORMAL)
            continue;
         any32 |= I->src[s].size == AGX_SIZE_32;
         any16 |= I->src[s].size == AGX_SIZE_16;
      }
      const bool dest16 = I->dest[0].size == AGX_SIZE_16;
      const bool cmp_mixed =
         select && I->src[0].size != I->src[1].size;
      /* Half selects are few and patchy on G17G (no LE/GE register form,
       * for one); the 32-bit register select is witnessed for eq/lt/gt/ge/le.
       * On this second attempt run every half-destination select there. */
      const bool sel_mixed =
         select && (dest16 ||
                    I->src[2].size != I->dest[0].size ||
                    I->src[3].size != I->dest[0].size ||
                    I->src[0].size != I->dest[0].size);
      const bool alu_mixed =
         !select && any16 && (any32 || !dest16);
      if (!cmp_mixed && !sel_mixed && !alu_mixed)
         continue;

      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         if (I->src[s].type != AGX_INDEX_NORMAL ||
             I->src[s].size != AGX_SIZE_16)
            continue;
         /* The 32-bit select compares f32: widen half comparands. */
         if (select && s < 2 && !cmp_mixed && !sel_mixed)
            continue;
         /* Selected values are bit patterns: zero-extend them, as the
          * existing G17G half-select path does. Floats convert (exact). */
         if (select && s >= 2) {
            if (I->src[s].abs || I->src[s].neg)
               continue;
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            agx_mov_to(&b, wide, I->src[s]);
            I->src[s] = wide;
            progress = true;
            continue;
         }
         agx_index plain = I->src[s];
         plain.abs = false;
         plain.neg = false;
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide, plain);
         wide.abs = I->src[s].abs;
         wide.neg = I->src[s].neg;
         I->src[s] = wide;
         progress = true;
      }

      if (dest16 && (alu_mixed || sel_mixed)) {
         agx_index narrow = I->dest[0];
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         I->dest[0] = wide;
         agx_builder after = agx_init_builder(ctx, agx_after_instr(I));
         /* A select keeps the low half of its zero-extended bits; an
          * arithmetic result converts once. */
         if (select)
            agx_subdivide_to(&after, narrow, wide, 0);
         else
            agx_fmov_to(&after, narrow, wide);
         progress = true;
      }
   }
   return progress;
}

/* The native half-width branch encoder supports register/register and
 * register/zero comparisons. Materialize other constants while SSA temporaries
 * can still be allocated; never invent a scratch register in the packer. */
static bool
agx_lower_agx3_cf_constants(agx_context *ctx)
{
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       ctx->key->dev.emit_agx3d) {
      bool progress = false;
      agx_foreach_instr_global_safe(ctx, I) {
         /* G18's branch compares registers of the same float width. A half
          * propagated into a float comparison must be widened explicitly,
          * after copy propagation and before RA. Binary16 to binary32 is
          * exact; keep the original condition and inversion unchanged. */
         if ((I->op == AGX_OPCODE_IF_FCMP ||
              I->op == AGX_OPCODE_WHILE_FCMP ||
              I->op == AGX_OPCODE_BREAK_IF_FCMP) && I->nr_srcs == 2) {
            for (unsigned s = 0; s < 2; ++s) {
               if (I->src[s].type != AGX_INDEX_NORMAL ||
                   I->src[s].size != AGX_SIZE_16 ||
                   I->src[s ^ 1].size != AGX_SIZE_32)
                  continue;
               agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, wide, I->src[s]);
               I->src[s] = wide;
               progress = true;
            }
            continue;
         }

         switch (I->op) {
         case AGX_OPCODE_IF_ICMP:
         case AGX_OPCODE_ELSE_ICMP:
         case AGX_OPCODE_WHILE_ICMP:
         case AGX_OPCODE_BREAK_IF_ICMP:
            break;
         default:
            continue;
         }
         if (I->nr_srcs != 2)
            continue;

         /* G18G has a grounded half register/register comparison. Its
          * immediate form only accepts zero, so allocate other constants
          * before RA without changing operand order or the predicate. */
         for (unsigned s = 0; s < 2; ++s) {
            agx_index src = I->src[s], other = I->src[1 - s];
            if (src.type != AGX_INDEX_IMMEDIATE || src.value == 0 ||
                src.value > UINT8_MAX || src.abs || src.neg ||
                other.type == AGX_INDEX_IMMEDIATE ||
                other.size != AGX_SIZE_16 || agx_channels(other) != 1)
               continue;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            I->src[s] = agx_mov_imm(&b, 16, src.value);
            progress = true;
         }
      }
      return progress;
   }

   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      bool floating = false;
      switch (I->op) {
      case AGX_OPCODE_IF_FCMP:
      case AGX_OPCODE_WHILE_FCMP:
      case AGX_OPCODE_BREAK_IF_FCMP:
         floating = true;
         break;
      case AGX_OPCODE_IF_ICMP:
      case AGX_OPCODE_ELSE_ICMP:
      case AGX_OPCODE_WHILE_ICMP:
      case AGX_OPCODE_BREAK_IF_ICMP:
         break;
      default:
         continue;
      }
      if (I->nr_srcs != 2)
         continue;
      /* Constant propagation may leave a constant predicate inside
       * structured control flow. The native integer branch still requires
       * one register comparator; keep the branch's mask-stack semantics.
       */
      if (!floating && I->src[0].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          !I->src[0].abs && !I->src[0].neg) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[0] = agx_mov_imm(&b, 32, I->src[0].value);
         progress = true;
      }
      /* Native FCMP branches compare full floats. Widen actual half
       * operands after copy propagation, which could otherwise fold this
       * conversion back into the branch. Binary16 values are exact in F32.
       * Immediate zero keeps the separately supported register/zero form. */
      if (floating) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].size != AGX_SIZE_16)
               continue;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, wide, I->src[s]);
            I->src[s] = wide;
            progress = true;
         }
      }
      /* Native branch predicates have no qualified absolute-value operand.
       * Clear the IEEE sign bit before RA, preserving zeros, infinities and
       * NaN payloads. If abs is followed by negation, materialize that bit too
       * so the register/zero predicate remains a plain operand. Negated
       * equality and unordered relations likewise need plain operands;
       * only strict LT/GT currently have a native negated encoding. */
      if (floating) {
         for (unsigned s = 0; s < 2; ++s) {
            agx_index src = I->src[s];
            const bool native_neg =
               (I->fcond == AGX_FCOND_LT || I->fcond == AGX_FCOND_GT) &&
               !I->src[0].abs && !I->src[1].abs &&
               I->src[0].type != AGX_INDEX_IMMEDIATE &&
               I->src[1].type != AGX_INDEX_IMMEDIATE &&
               (I->src[0].neg != I->src[1].neg);
            if ((!src.abs && (!src.neg || native_neg)) ||
                src.type == AGX_INDEX_IMMEDIATE ||
                src.size != AGX_SIZE_32 || agx_channels(src) != 1)
               continue;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            const bool negate = src.neg, clear_sign = src.abs;
            src.abs = src.neg = false;
            agx_index absolute = src;
            if (clear_sign) {
               absolute = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, absolute, src,
                            agx_mov_imm(&b, 32, 0x7fffffff), 0x8);
            }
            if (negate) {
               agx_index negative = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, negative, absolute,
                            agx_mov_imm(&b, 32, 0x80000000), 0x6);
               absolute = negative;
            }
            I->src[s] = absolute;
            progress = true;
         }
      }
      for (unsigned s = 0; s < 2; ++s) {
         agx_index src = I->src[s], other = I->src[1 - s];
         if (floating) {
            /* FCMP has a register/zero form, but arbitrary minifloats must
             * use its register/register form. Preserve IEEE value and source
             * modifiers while allocating the temporary through normal SSA. */
            if (src.type != AGX_INDEX_IMMEDIATE || src.value == 0 ||
                src.value > UINT8_MAX || other.type == AGX_INDEX_IMMEDIATE ||
                other.size != AGX_SIZE_32 || agx_channels(other) != 1)
               continue;
            float value = agx_minifloat_decode(src.value);
            if (src.abs)
               value = fabsf(value);
            if (src.neg)
               value = -value;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            I->src[s] = agx_mov_imm(&b, 32, fui(value));
            progress = true;
            continue;
         }
         if (src.type != AGX_INDEX_IMMEDIATE || src.value == 0 ||
             src.value > UINT8_MAX || src.abs || src.neg ||
             other.type == AGX_INDEX_IMMEDIATE || other.size != AGX_SIZE_16 ||
             agx_channels(other) != 1)
            continue;
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[s] = agx_mov_imm(&b, 16, src.value);
         progress = true;
      }
   }
   return progress;
}

/* Half-width shift-add has register operands. Fold an immediate's integer
 * negation modulo 2^16 and allocate it normally before RA. This also covers
 * decrementing 16-bit loop counters without guessing a half subtract opcode. */
static bool
agx_lower_agx3_half_iadd_constants(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      /* 64-bit adds have register operands too: no grounded form takes an
       * immediate (float_controls.fp32.input_args.rounding_rte_conv_from_
       * {int,uint}64_*_vert, whose constants stay immediates in vertex
       * shaders). */
      if (I->op != AGX_OPCODE_IADD || I->nr_dests != 1 || I->nr_srcs != 2 ||
          (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_64) ||
          agx_channels(I->dest[0]) != 1 || I->saturate)
         continue;
      const bool wide = I->dest[0].size == AGX_SIZE_64;
      agx_foreach_src(I, s) {
         agx_index src = I->src[s];
         if (src.type != AGX_INDEX_IMMEDIATE || src.value > UINT8_MAX || src.abs)
            continue;
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         if (wide) {
            uint64_t value = src.neg ? -(uint64_t)src.value : src.value;
            I->src[s] = agx_mov_imm(&b, 64, value);
         } else {
            uint16_t value = src.neg ? -src.value : src.value;
            I->src[s] = agx_mov_imm(&b, 16, value);
         }
         progress = true;
      }
   }
   return progress;
}

/* A 32-bit add or multiply-add of halves -- 16-bit integer arithmetic the
 * optimizer widened, as hk's int16 shaders and dynamically indexed uniform
 * arrays produce -- has no mixed-width AGX3 form. Extend each half register
 * explicitly: AGX IR plain half sources are signed and abs selects zero
 * extension; negation and the add's shift stay on the widened operation. */
static bool
agx_lower_agx3_widening_int(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (!((I->op == AGX_OPCODE_IMAD && I->nr_srcs == 3) ||
            (I->op == AGX_OPCODE_IADD && I->nr_srcs == 2 && !I->saturate)) ||
          I->nr_dests != 1 || I->dest[0].size != AGX_SIZE_32 ||
          agx_channels(I->dest[0]) != 1)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_foreach_src(I, s) {
         agx_index src = I->src[s];
         if (src.size != AGX_SIZE_16 || src.type == AGX_INDEX_IMMEDIATE ||
             src.type == AGX_INDEX_NULL || agx_channels(src) != 1)
            continue;
         bool is_unsigned = src.abs;
         bool negate = src.neg;
         src.abs = false;
         src.neg = false;
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         if (is_unsigned)
            agx_mov_to(&b, wide, src);
         else
            agx_signext_to(&b, wide, src);
         I->src[s] = negate ? agx_neg(wide) : wide;
         progress = true;
      }
   }
   return progress;
}

/* The immediate native descriptor index has four bits. Materialize larger
 * indices as the existing zero-extended 64-bit indexed-fetch operand after
 * copy propagation, so constants do not get folded back into that field. */
static bool
agx3_lower_large_fetch_indices(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return false;
   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_TEXTURE_LOAD || I->nr_srcs != 6)
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      if (I->lod_mode == AGX_LOD_MODE_LOD_MIN &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          (I->src[1].value > 31 || I->src[1].neg)) {
         uint32_t level = I->src[1].neg ? -(uint32_t)I->src[1].value
                                      : I->src[1].value;
         I->src[1] = agx_mov_imm(&b, 32, level);
         progress = true;
      }
      if (I->src[2].type == AGX_INDEX_IMMEDIATE && I->src[2].value == 0 &&
          I->src[3].type == AGX_INDEX_IMMEDIATE && I->src[3].value > 15) {
         I->src[3] = agx_mov_imm(&b, 64, I->src[3].value);
         progress = true;
      }
   }
   return progress;
}

/* No grounded AGX3 form adds two 64-bit registers (address forms take one
 * 64-bit base). Split a plain 64-bit add into a low 32-bit add, a carry from
 * an unsigned compare and a high 32-bit add, then recombine the pair
 * (float_controls.fp32.input_args.rounding_rte_conv_from_{int,uint}64_*). */
static bool
agx_lower_agx3_wide_iadd(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_IADD || I->nr_dests != 1 || I->nr_srcs != 2 ||
          I->dest[0].size != AGX_SIZE_64 || agx_channels(I->dest[0]) != 1 ||
          I->shift != 0 || I->saturate)
         continue;
      bool wide_regs = true;
      agx_foreach_src(I, s) {
         if (I->src[s].type != AGX_INDEX_NORMAL || I->src[s].size != AGX_SIZE_64 ||
             I->src[s].abs || I->src[s].neg || agx_channels(I->src[s]) != 1)
            wide_regs = false;
      }
      if (!wide_regs)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index lo[2], hi[2];
      for (unsigned s = 0; s < 2; ++s) {
         lo[s] = agx_temp(ctx, AGX_SIZE_32);
         hi[s] = agx_temp(ctx, AGX_SIZE_32);
         agx_subdivide_to(&b, lo[s], I->src[s], 0);
         agx_subdivide_to(&b, hi[s], I->src[s], 1);
      }
      agx_index sum_lo = agx_iadd(&b, lo[0], lo[1], 0);
      agx_index carry =
         agx_icmpsel(&b, sum_lo, lo[0], agx_mov_imm(&b, 32, 1),
                     agx_mov_imm(&b, 32, 0), AGX_ICOND_ULT);
      agx_index sum_hi =
         agx_iadd(&b, agx_iadd(&b, hi[0], hi[1], 0), carry, 0);
      /* Two 32-bit halves collect straight into the 64-bit value, as
       * nir_op_pack_64_2x32_split does. */
      agx_emit_collect_to(&b, I->dest[0], 2, (agx_index[]){sum_lo, sum_hi});
      agx_remove_instruction(I);
      progress = true;
   }
   return progress;
}

/* The exact operand shape of the packer's
 * "bitwise operation against an IMMEDIATE" rule (agx3_pack_binary.c): one
 * AND/OR/XOR truth table, a single 8-bit non-modified immediate and one
 * width-matched scalar SSA value that register allocation turns into a GPR.
 * Uniform or immediate partners keep the old materialization. */
static bool
agx_agx3_bitop_immediate_ok(const agx_context *ctx, const agx_instr *I)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       I->op != AGX_OPCODE_BITOP || I->nr_dests != 1 || I->nr_srcs != 2 ||
       I->saturate || agx_channels(I->dest[0]) != 1 ||
       (I->dest[0].size != AGX_SIZE_16 && I->dest[0].size != AGX_SIZE_32))
      return false;

   const unsigned tt = I->truth_table & 0xfu;
   if (tt != 0x8 /* AND */ && tt != 0xe /* OR */ && tt != 0x6 /* XOR */)
      return false;

   const bool imm0 = I->src[0].type == AGX_INDEX_IMMEDIATE;
   const bool imm1 = I->src[1].type == AGX_INDEX_IMMEDIATE;
   if (imm0 == imm1)
      return false;

   const agx_index imm = I->src[imm0 ? 0 : 1];
   const agx_index reg = I->src[imm0 ? 1 : 0];
   return imm.value <= 0xff && !imm.abs && !imm.neg && !imm.memory &&
          reg.type == AGX_INDEX_NORMAL && !reg.memory && !reg.abs &&
          !reg.neg && agx_channels(reg) == 1 && reg.size == I->dest[0].size;
}

void
agx_materialize_agx3_alu_constants(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

      /* G18G's mixed half-compare/full-result ISEL admits one selected
       * immediate. Keep the compare intact and materialize the other value.
       * M6 only: the form is grounded by G18G witnesses alone. */
      if (agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_ICMPSEL && I->nr_dests == 1 &&
          I->nr_srcs == 4 && !I->saturate &&
          I->dest[0].size == AGX_SIZE_32 &&
          agx_channels(I->dest[0]) == 1 &&
          (I->icond == AGX_ICOND_UEQ || I->icond == AGX_ICOND_SEQ) &&
          I->src[0].type == AGX_INDEX_NORMAL &&
          I->src[1].type == AGX_INDEX_NORMAL &&
          I->src[0].size == AGX_SIZE_16 &&
          I->src[1].size == AGX_SIZE_16 &&
          I->src[2].type == AGX_INDEX_IMMEDIATE &&
          I->src[3].type == AGX_INDEX_IMMEDIATE &&
          !I->src[2].abs && !I->src[2].neg &&
          !I->src[3].abs && !I->src[3].neg &&
          I->src[2].value <= UINT8_MAX && I->src[3].value <= UINT8_MAX) {
         I->src[3] = agx_mov_imm(&b, 32, I->src[3].value);
      }

      /* A half source feeding a 32-bit square-root step is widened exactly
       * first. Apple's half-input step (agx3_alu_sqrt_step_half) is grounded
       * only with a cached source and result, so the packer refuses the other
       * forms. G17P keeps that native step, and its existing output. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          ctx->key->dev.chip != AGX_CHIP_G17P &&
          I->op == AGX_OPCODE_SRSQRT && I->nr_srcs == 1 &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[0].size == AGX_SIZE_16 &&
          I->src[0].type != AGX_INDEX_IMMEDIATE) {
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide, I->src[0]);
         I->src[0] = wide;
      }

      /* An AGX3 float add, multiply or fused multiply-add has no form whose
       * operands are all immediates. The optimizer can leave one after
       * folding a select to a constant; keep the first immediate and move
       * the others into registers. */
      bool all_immediate = I->nr_srcs > 0;
      agx_foreach_src(I, s)
         all_immediate &= I->src[s].type == AGX_INDEX_IMMEDIATE;
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B && all_immediate &&
          (I->op == AGX_OPCODE_FADD || I->op == AGX_OPCODE_FMUL ||
           I->op == AGX_OPCODE_FFMA)) {
         agx_foreach_src(I, s) {
            if (s == 0)
               continue;
            float value = agx_minifloat_decode(I->src[s].value);
            if (I->src[s].abs)
               value = fabsf(value);
            if (I->src[s].neg)
               value = -value;
            I->src[s] = I->dest[0].size == AGX_SIZE_16
                           ? agx_mov_imm(&b, 16, _mesa_float_to_half(value))
                           : agx_mov_imm(&b, 32, fui(value));
         }
      }

      /* AGX3's integer compare-select names an immediate comparand in eight
       * bits (agx3_pack_binary.c). A wider constant goes through a register
       * of the other comparand's width. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (I->op == AGX_OPCODE_ICMPSEL || I->op == AGX_OPCODE_ICMP)) {
         for (unsigned s = 0; s < 2; ++s) {
            const agx_index other = I->src[s ^ 1];
            if (I->src[s].type != AGX_INDEX_IMMEDIATE ||
                I->src[s].value <= 0xff ||
                other.type == AGX_INDEX_IMMEDIATE)
               continue;
            I->src[s] = agx_mov_imm(
               &b, other.size == AGX_SIZE_16 ? 16 : 32, I->src[s].value);
         }
      }

      /* XNOR with a constant is XOR with the complemented constant. Native
       * G17G uses the same identity for ushort masks, sourcing the resulting
       * full-width mask from its constant program. Materialize before RA so
       * the XOR's register operand and lifetime are allocated normally. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          I->op == AGX_OPCODE_BITOP && I->nr_dests == 1 && I->nr_srcs == 2 &&
          !I->saturate && I->dest[0].size == AGX_SIZE_16 &&
          (I->truth_table & 0xf) == 9) {
         for (unsigned i = 0; i < 2; ++i) {
            const agx_index imm = I->src[i];
            if (imm.type != AGX_INDEX_IMMEDIATE || imm.abs || imm.neg ||
                imm.memory)
               continue;
            I->src[i] = agx_mov_imm(&b, 16, (~imm.value) & 0xffff);
            I->truth_table = 6; /* XOR */
            break;
         }
      }

      /* A general 32-bit FADD of a half and an immediate has no single
       * AGX3C form. G17G converts first (f32add of the half and -0.0) and
       * then adds in f32: saturate(float(h)) is f32add, f32add.sat
       * (port/hfwit k_sat_h2f, k_h2f_mul). Widen the half the same exact
       * way. The exception is that conversion itself, an unsaturated
       * f32add of the half and -0.0, which is native: widening it again
       * would convert the same value twice (every float(half) operand of an
       * FMA paid two). */
      const bool half_to_float =
         I->op == AGX_OPCODE_FADD && I->nr_srcs == 2 && !I->saturate &&
         I->src[1].type == AGX_INDEX_IMMEDIATE &&
         agx_is_equiv(I->src[1], agx_negzero()) && !I->src[1].abs &&
         !I->src[1].neg && I->src[0].type != AGX_INDEX_IMMEDIATE;
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          I->op == AGX_OPCODE_FADD && I->nr_dests == 1 && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_32 && !half_to_float &&
          (I->src[0].type == AGX_INDEX_IMMEDIATE ||
           I->src[1].type == AGX_INDEX_IMMEDIATE)) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].type == AGX_INDEX_UNIFORM ||
                I->src[s].size != AGX_SIZE_16)
               continue;
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, wide, I->src[s]);
            I->src[s] = wide;
         }
      }

      /* The shared 32-bit saturating IADD path takes two registers.
       * Robust buffer bounds checks naturally produce uadd_sat(offset, #size),
       * where an immediate is implicitly unsigned and therefore lacks the ABS
       * modifier carried by the register operand. Materialize the immediate
       * before RA and copy the operation's signedness onto the new register so
       * the existing register/register iadd_sat/uadd_sat encoding applies. */
      if (I->op == AGX_OPCODE_IADD && I->saturate && I->nr_dests == 1 &&
          I->nr_srcs == 2 && I->dest[0].size == AGX_SIZE_32 &&
          (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
           (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
            (I->src[0].abs || I->src[1].abs)))) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type != AGX_INDEX_IMMEDIATE)
               continue;

            const bool negate = I->src[s].neg;
            const bool absolute = I->src[s].abs || I->src[s ^ 1].abs;
            agx_index value = agx_mov_imm(&b, 32, I->src[s].value);
            value.abs = absolute;
            value.neg = negate;
            I->src[s] = value;
         }
      }

      /* A half fabs is initially represented as fadd(abs(x), -0.0h).  The
       * G17P half-add form has no absolute-source field, and leaving this
       * pseudo operation intact until after register allocation makes the
       * packer refuse frexp/modf precision shaders.  fabs on an IEEE half is
       * exactly a sign-bit clear, so lower that exact move to the grounded
       * half BITOP path while the mask can still receive an SSA register. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FADD && I->nr_dests == 1 && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_16 && !I->saturate &&
          I->src[0].size == AGX_SIZE_16 && I->src[0].abs &&
          !I->src[0].neg &&
          agx_is_equiv(I->src[1], agx_immediate(UINT8_C(0x80)))) {
         agx_index source = I->src[0];
         source.abs = false;
         agx_index mask = agx_mov_imm(&b, 16, UINT16_C(0x7fff));

         I->op = AGX_OPCODE_BITOP;
         I->src[0] = source;
         I->src[1] = mask;
         I->truth_table = 0x8; /* AND */
      }

      /* The grounded scalar-half FSEL and rounding forms do not carry float
       * source modifiers. CTS reaches both through frexp/modf, with fabs
       * folded into a comparator or floor input. Preserve every half bit by
       * materializing abs/neg as sign-bit operations before RA instead of
       * refusing the post-RA instruction. */
      const bool half_fsel =
         agx_isa_is_agx3(ctx->key->dev.encoding) &&
         I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
         (I->dest[0].size == AGX_SIZE_16 ||
          (I->dest[0].size == AGX_SIZE_32 &&
           I->src[0].size == AGX_SIZE_16 &&
           I->src[1].size == AGX_SIZE_16));
      const bool half_round =
         agx_isa_is_agx3(ctx->key->dev.encoding) &&
         (I->op == AGX_OPCODE_FLOOR || I->op == AGX_OPCODE_CEIL) &&
         I->nr_srcs == 1 && I->dest[0].size == AGX_SIZE_16;
      /* The modified add form needs a 32-bit destination, so a half add of
       * -x or |x| -- mediump subtraction, e.g. range reduction in trig --
       * has no encoding either. */
      /* Leave an immediate beside the modifier alone: #1.0 - x has its own
       * witnessed half form (agx3_alu_hadd_one_neg) and a plain x + #imm
       * does not. */
      const bool half_add_candidate =
         agx_isa_is_agx3(ctx->key->dev.encoding) &&
         I->op == AGX_OPCODE_FADD && I->nr_srcs == 2 && !I->saturate &&
         I->dest[0].size == AGX_SIZE_16;
      const int add_imm = !half_add_candidate                       ? -1
                          : I->src[0].type == AGX_INDEX_IMMEDIATE ? 0
                          : I->src[1].type == AGX_INDEX_IMMEDIATE ? 1
                                                                  : -1;
      const bool one_minus =
         add_imm >= 0 && I->src[add_imm].value == UINT8_C(0x30) &&
         !I->src[add_imm].abs && !I->src[add_imm].neg &&
         I->src[add_imm ^ 1].type != AGX_INDEX_IMMEDIATE &&
         I->src[add_imm ^ 1].neg && !I->src[add_imm ^ 1].abs;
      /* Apart from that and the signed-zero adds that implement
       * conversions, a half add has no immediate form, with or without a
       * modifier beside it -- mediump c - x and x - c, e.g. dEQP's trig range
       * reduction. Make the immediate an exact half register; a modifier is
       * made explicit below, and the add becomes a register HADD. */
      const bool signed_zero =
         add_imm >= 0 && (I->src[add_imm].value & 0x7f) == 0;
      if (half_add_candidate && add_imm >= 0 && !one_minus && !signed_zero &&
          I->src[add_imm ^ 1].type != AGX_INDEX_IMMEDIATE &&
          I->src[add_imm ^ 1].size == AGX_SIZE_16) {
         float value = agx_minifloat_decode(I->src[add_imm].value);
         if (I->src[add_imm].abs)
            value = fabsf(value);
         if (I->src[add_imm].neg)
            value = -value;
         I->src[add_imm] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
      }
      const bool half_add =
         half_add_candidate &&
         I->src[0].type != AGX_INDEX_IMMEDIATE &&
         I->src[1].type != AGX_INDEX_IMMEDIATE;

      /* A modified half multiplier beside a literal has no native form.
       * Keep half multiplication and its rounding, but use two registers
       * with the exact sign-bit operations shared by the half add/select. */
      const bool half_mul =
         agx_isa_is_agx3(ctx->key->dev.encoding) &&
         I->op == AGX_OPCODE_HMUL && I->nr_srcs == 2 && !I->saturate &&
         I->dest[0].size == AGX_SIZE_16 &&
         ((I->src[0].type == AGX_INDEX_IMMEDIATE &&
           I->src[1].size == AGX_SIZE_16 &&
           (I->src[1].abs || I->src[1].neg)) ||
          (I->src[1].type == AGX_INDEX_IMMEDIATE &&
           I->src[0].size == AGX_SIZE_16 &&
           (I->src[0].abs || I->src[0].neg)));
      if (half_mul) {
         const unsigned imm = I->src[0].type == AGX_INDEX_IMMEDIATE ? 0 : 1;
         float value = agx_minifloat_decode(I->src[imm].value);
         if (I->src[imm].abs)
            value = fabsf(value);
         if (I->src[imm].neg)
            value = -value;
         I->src[imm] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
      }

      if (half_fsel || half_round || half_add || half_mul) {
         const unsigned source_count = half_fsel ? 4
                                       : half_add || half_mul ? 2 : 1;
         for (unsigned s = 0; s < source_count; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].size != AGX_SIZE_16 ||
                (!I->src[s].abs && !I->src[s].neg))
               continue;

            const bool absolute = I->src[s].abs;
            const bool negate = I->src[s].neg;
            agx_index value = I->src[s];
            value.abs = false;
            value.neg = false;

            if (absolute) {
               agx_index mask = agx_mov_imm(&b, 16, UINT16_C(0x7fff));
               agx_index result = agx_temp(ctx, AGX_SIZE_16);
               agx_bitop_to(&b, result, value, mask, 0x8 /* AND */);
               value = result;
            }

            if (negate) {
               agx_index sign = agx_mov_imm(&b, 16, UINT16_C(0x8000));
               agx_index result = agx_temp(ctx, AGX_SIZE_16);
               agx_bitop_to(&b, result, value, sign, 0x6 /* XOR */);
               value = result;
            }

            I->src[s] = value;
         }

         /* Both sources are now plain halves. The register/register half add
          * is encoded as HADD, the op NIR's own 16-bit fadd produces; a
          * 16-bit FADD that reaches the packer has no form. */
         if (half_add && I->src[0].size == AGX_SIZE_16 &&
             I->src[1].size == AGX_SIZE_16)
            I->op = AGX_OPCODE_HADD;
      }

      /* frexp's full-precision path compares zero against fabs(x) and
       * selects two full floats.  G17P's zero-comparison FSEL is grounded
       * only without source modifiers, so make fabs/fneg exact with integer
       * sign-bit operations while the modified comparator is still SSA. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32) {
         const bool zero_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE && I->src[0].value == 0;
         const bool zero_second =
            I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0;

         if (zero_first != zero_second) {
            const unsigned s = zero_first ? 1 : 0;
            if (I->src[s].type != AGX_INDEX_IMMEDIATE &&
                I->src[s].size == AGX_SIZE_32 &&
                (I->src[s].abs || I->src[s].neg)) {
               const bool absolute = I->src[s].abs;
               const bool negate = I->src[s].neg;
               agx_index value = I->src[s];
               value.abs = false;
               value.neg = false;

               if (absolute) {
                  agx_index mask =
                     agx_mov_imm(&b, 32, UINT32_C(0x7fffffff));
                  agx_index result = agx_temp(ctx, AGX_SIZE_32);
                  agx_bitop_to(&b, result, value, mask, 0x8 /* AND */);
                  value = result;
               }

               if (negate) {
                  agx_index sign =
                     agx_mov_imm(&b, 32, UINT32_C(0x80000000));
                  agx_index result = agx_temp(ctx, AGX_SIZE_32);
                  agx_bitop_to(&b, result, value, sign, 0x6 /* XOR */);
                  value = result;
               }

               I->src[s] = value;
            }
         }
      }
      /* Chromium/Skia compares a negated f32 register for equality and then
       * selects two unrelated f32 registers. The only native FSEL negate bit
       * is grounded for a different half-mask endpoint, so do not infer that
       * modifier across result shapes. Materialize IEEE fneg exactly by
       * flipping the sign bit, using the already-grounded MOV-immediate and
       * BITOP XOR forms, before register allocation. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 && !I->invert_cond && !I->saturate &&
          I->fcond == AGX_FCOND_EQ && I->src[0].size == AGX_SIZE_32 &&
          I->src[0].type != AGX_INDEX_IMMEDIATE && I->src[0].neg &&
          !I->src[0].abs && I->src[1].size == AGX_SIZE_32 &&
          I->src[1].type != AGX_INDEX_IMMEDIATE && !I->src[1].abs &&
          !I->src[1].neg && I->src[2].size == AGX_SIZE_32 &&
          I->src[2].type != AGX_INDEX_IMMEDIATE && !I->src[2].abs &&
          !I->src[2].neg && I->src[3].size == AGX_SIZE_32 &&
          I->src[3].type != AGX_INDEX_IMMEDIATE && !I->src[3].abs &&
          !I->src[3].neg) {
         agx_index source = I->src[0];
         source.neg = false;
         agx_index sign = agx_mov_imm(&b, 32, UINT32_C(0x80000000));
         agx_index flipped = agx_temp(ctx, AGX_SIZE_32);
         agx_bitop_to(&b, flipped, source, sign, 0x6 /* XOR */);
         I->src[0] = flipped;
      }

      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_BITREV && I->src[0].size == AGX_SIZE_16) {
         /* Copy propagation may narrow an otherwise word-wide reversal. */
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_mov_to(&b, wide, I->src[0]);
         I->src[0] = wide;
      }
      /* ISEL accepts a register comparator and an immediate comparator,
       * including ICMP pseudos lowered to ISEL after register allocation.
       */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          (I->op == AGX_OPCODE_ICMP || I->op == AGX_OPCODE_ICMPSEL) &&
          I->src[0].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          !I->src[0].abs && !I->src[0].neg)
         I->src[0] = agx_mov_imm(&b, 32, I->src[0].value);
      /* A negated full-width IADD source means subtraction, not signed
       * extension. Materialize the minuend so the native register ISUB can
       * implement constant - value, including the negation in udiv refinement.
       * Shifted subtraction has already been lowered to a separate shift. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_IADD && I->nr_dests == 1 && I->nr_srcs == 2 &&
          !I->saturate && !I->shift && I->dest[0].size == AGX_SIZE_32 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE && !I->src[0].abs &&
          !I->src[0].neg && I->src[1].size == AGX_SIZE_32 &&
          I->src[1].neg && !I->src[1].abs &&
          I->src[1].type != AGX_INDEX_IMMEDIATE) {
         I->src[0] = agx_mov_imm(&b, 32, I->src[0].value);
      }
      /* SIGNEXT becomes a signed half IADD on AGX2. Keep the AGX3 lowering
       * in SSA, where temporary registers are still available: preserve the
       * low 16 bits, move the sign bit to bit 31, then shift arithmetically.
       */
      if (I->op == AGX_OPCODE_SIGNEXT &&
          ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_mov_to(&b, wide, I->src[0]);
         agx_index shifted = agx_temp(ctx, AGX_SIZE_32);
         agx_bfi_to(&b, shifted, agx_zero(), wide,
                    agx_mov_imm(&b, 32, 16), 0);
         agx_asr_to(&b, I->dest[0], shifted, agx_immediate(16));
         agx_remove_instruction(I);
         continue;
      }
      /* G18G: copy propagation may leave a half source on a full-result RCP.
       * Preserve that value by widening before RA, keeping the native
       * full-precision reciprocal and its result width unchanged. */
      if (agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_RCP && I->nr_dests == 1 && I->nr_srcs == 1 &&
          I->dest[0].size == AGX_SIZE_32 && !I->saturate &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[0].size == AGX_SIZE_16 && !I->src[0].memory &&
          !I->src[0].abs && !I->src[0].neg) {
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide, I->src[0]);
         I->src[0] = wide;
      }

      /* G18G: FCMP becomes a boolean FSEL after register allocation. Legalize
       * its half/half comparison while a temporary can still be allocated;
       * the native mixed comparator form keeps the integer0/1 result. */
      if (agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_FCMP && I->nr_dests == 1 && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_16 && !I->saturate &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].type != AGX_INDEX_IMMEDIATE &&
          I->src[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16 &&
          !I->src[0].memory && !I->src[1].memory &&
          !I->src[0].abs && !I->src[0].neg &&
          !I->src[1].abs && !I->src[1].neg) {
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide, I->src[0]);
         I->src[0] = wide;
      }

      /* A native G17G/G18G half-result FSEL accepts a full comparator
       * followed by a half comparator and a half-register/zero choice.
       * Widening one binary16 comparator is exact and leaves the selected
       * half bit patterns and every other use of the input unchanged. */
      if (agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_dests == 1 && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 && !I->saturate &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].type != AGX_INDEX_IMMEDIATE &&
          I->src[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16 &&
          !I->src[0].memory && !I->src[1].memory &&
          !I->src[0].abs && !I->src[0].neg &&
          !I->src[1].abs && !I->src[1].neg) {
         bool zero_half_choice = false;
         for (unsigned s = 2; s < 4; ++s) {
            const agx_index zero = I->src[s];
            const agx_index reg = I->src[s == 2 ? 3 : 2];
            zero_half_choice |= zero.type == AGX_INDEX_IMMEDIATE &&
               !zero.value && !zero.abs && !zero.neg && !zero.memory &&
               reg.type != AGX_INDEX_IMMEDIATE && reg.size == AGX_SIZE_16 &&
               !reg.abs && !reg.neg && !reg.memory;
         }
         /* FSEL's selected immediates are integer bit patterns. The
          * native mixed-comparator half-result form also accepts an exact
          * boolean pair, so preserve those literals without conversion. */
         const agx_index yes = I->src[2], no = I->src[3];
         const bool boolean_choice =
            yes.type == AGX_INDEX_IMMEDIATE &&
            no.type == AGX_INDEX_IMMEDIATE &&
            yes.value <= 1 && no.value <= 1 && yes.value + no.value == 1 &&
            !yes.abs && !yes.neg && !yes.memory &&
            !no.abs && !no.neg && !no.memory;
         if (zero_half_choice || boolean_choice) {
            agx_index wide = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, wide, I->src[0]);
            I->src[0] = wide;
         }
      }

      /* G18G's full-result register/zero FSEL requires full comparators.
       * Constant folding can leave an exactly representable half constant
       * in either compare slot. Widen those operands before allocation;
       * selected values retain their integer bit-pattern semantics. Keep
       * the separately grounded all-immediate boolean forms unchanged. */
      if (agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_dests == 1 && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 && !I->saturate) {
         bool register_zero_select = true;
         bool selected_register = false;
         for (unsigned s = 2; s < 4; ++s) {
            const agx_index src = I->src[s];
            const bool reg = src.type != AGX_INDEX_IMMEDIATE &&
                             src.size == AGX_SIZE_32 && !src.memory;
            const bool zero = src.type == AGX_INDEX_IMMEDIATE && !src.value;
            register_zero_select &= (reg || zero) && !src.abs && !src.neg;
            selected_register |= reg;
         }
         if (register_zero_select && selected_register) {
            for (unsigned c = 0; c < 2; ++c) {
               if (I->src[c].type == AGX_INDEX_IMMEDIATE ||
                   I->src[c].size != AGX_SIZE_16 || I->src[c].memory)
                  continue;
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, wide, I->src[c]);
               I->src[c] = wide;
            }
         }
      }

      /* AGX3C's tied min/max consumes full register operands in both
       * positions. Keep constants tied after compact-constant folding. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 && !I->invert_cond &&
          (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN)) {
         for (unsigned c = 0; c < 2; ++c) {
            if (I->src[c].type != AGX_INDEX_IMMEDIATE ||
                !agx_is_equiv(I->src[c], I->src[c + 2]))
               continue;
            const agx_index constant = agx_mov_imm(
               &b, 32, fui(agx_minifloat_decode(I->src[c].value)));
            I->src[c] = I->src[c + 2] = constant;
         }
      }

      /* General FSEL has register comparators or an implicit zero. Nonzero
       * minifloats in compare slots must first become floating registers;
       * selected values in slots 2/3 have integer bit-pattern semantics.
       * FCMP is lowered to a boolean FSEL later, so normalize it here too.
       */
      if (ctx->key->dev.chip != AGX_CHIP_G17P &&
          (I->op == AGX_OPCODE_FCMPSEL || I->op == AGX_OPCODE_FCMP)) {
         if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
             I->op == AGX_OPCODE_FCMP) {
            /* Boolean FCMP becomes an immediate-alternative FSEL after this
             * pass; that form also needs full-width comparators. */
            for (unsigned s = 0; s < 2; ++s) {
               if (I->src[s].type != AGX_INDEX_IMMEDIATE &&
                   I->src[s].size == AGX_SIZE_16) {
                  agx_index wide = agx_temp(ctx, AGX_SIZE_32);
                  agx_fmov_to(&b, wide, I->src[s]);
                  I->src[s] = wide;
               }
            }
         }
         if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
             I->op == AGX_OPCODE_FCMPSEL && I->dest[0].size == AGX_SIZE_16) {
            /* Select alternatives are bit patterns, not minifloats -- except
             * in a tied minimum/maximum, where the optimizer deposited the
             * compared minifloat into the select slot too. */
            for (unsigned s = 2; s < 4; ++s) {
               if (I->src[s].type != AGX_INDEX_IMMEDIATE)
                  continue;
               const agx_index cmp = I->src[s - 2];
               const bool tied = cmp.type == AGX_INDEX_IMMEDIATE &&
                                 cmp.value == I->src[s].value &&
                                 cmp.abs == I->src[s].abs &&
                                 cmp.neg == I->src[s].neg;
               uint32_t bits = I->src[s].value;
               if (tied) {
                  float value = agx_minifloat_decode(I->src[s].value);
                  if (I->src[s].abs)
                     value = fabsf(value);
                  if (I->src[s].neg)
                     value = -value;
                  bits = _mesa_float_to_half(value);
               }
               I->src[s] = agx_mov_imm(&b, 16, bits);
            }
         }
         /* G18G's half FSEL is grounded only with all four sources in half
          * registers (agx3_alu_fcmp_select_half_gpu1). Compare slots hold
          * minifloats, so materialize their half value; select slots are
          * bit patterns. M6 only. */
         if (agx_is_g18g(&ctx->key->dev) &&
             I->op == AGX_OPCODE_FCMPSEL && I->dest[0].size == AGX_SIZE_16) {
            for (unsigned s = 0; s < 4; ++s) {
               agx_index v = I->src[s];
               if (v.type != AGX_INDEX_IMMEDIATE)
                  continue;
               if (s < 2) {
                  /* Match the width of the other compared value. */
                  const bool wide = I->src[1 - s].type != AGX_INDEX_IMMEDIATE &&
                                    I->src[1 - s].size == AGX_SIZE_32;
                  float f = agx_minifloat_decode(v.value);
                  if (v.abs)
                     f = fabsf(f);
                  if (v.neg)
                     f = -f;
                  I->src[s] = wide ? agx_mov_imm(&b, 32, fui(f))
                                   : agx_mov_imm(&b, 16, _mesa_float_to_half(f));
               } else {
                  I->src[s] = agx_mov_imm(&b, 16, v.value);
               }
            }
         }
         /* The native F32/F16 route is qualified for EQ/LT and two full
          * register alternatives. Other full-result shapes use the general
          * F32 comparator route. F16->F32 is exact, including signed zero;
          * retain the condition, inversion, and selected bit patterns.
          * AGX3B only: G17G selects 32-bit values on half comparands
          * directly (port/rawit k_fcmpsel_mixed, k_fcmpsel_mixed2). */
         if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             I->op == AGX_OPCODE_FCMPSEL && I->dest[0].size == AGX_SIZE_32) {
            const bool native_mixed =
               I->src[0].size == AGX_SIZE_32 &&
               I->src[0].type != AGX_INDEX_IMMEDIATE &&
               I->src[1].size == AGX_SIZE_16 &&
               I->src[1].type != AGX_INDEX_IMMEDIATE &&
               !I->invert_cond &&
               (I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT) &&
               I->src[2].type != AGX_INDEX_IMMEDIATE &&
               I->src[3].type != AGX_INDEX_IMMEDIATE &&
               I->src[2].size == AGX_SIZE_32 && I->src[3].size == AGX_SIZE_32;
            if (!native_mixed) {
               for (unsigned c = 0; c < 2; ++c) {
                  /* A uniform comparand declares its own width (f43/f60);
                   * agx3_alu_fcmp_select keeps a half uniform in place. */
                  if (I->src[c].type == AGX_INDEX_IMMEDIATE ||
                      I->src[c].type == AGX_INDEX_UNIFORM ||
                      I->src[c].size != AGX_SIZE_16)
                     continue;
                  agx_index wide = agx_temp(ctx, AGX_SIZE_32);
                  agx_fmov_to(&b, wide, I->src[c]);
                  I->src[c] = wide;
               }
            }
         }
         if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
             I->op == AGX_OPCODE_FCMPSEL &&
             I->dest[0].size == AGX_SIZE_16 &&
             !(agx_is_equiv(I->src[0], I->src[2]) &&
               agx_is_equiv(I->src[1], I->src[3]) &&
               !I->invert_cond &&
               (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN))) {
            /* Only the tied native half min/max form accepts two half
             * comparators. Widen general comparisons exactly, independently
             * of the bit-pattern width of their selected alternatives. */
            for (unsigned c = 0; c < 2; ++c) {
               if (I->src[c].type == AGX_INDEX_IMMEDIATE ||
                   I->src[c].type == AGX_INDEX_UNIFORM ||
                   I->src[c].size != AGX_SIZE_16)
                  continue;
               agx_index wide = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, wide, I->src[c]);
               I->src[c] = wide;
            }
         }
         const bool modified = I->src[0].abs || I->src[0].neg ||
                               I->src[1].abs || I->src[1].neg;
         for (unsigned s = 0; s < 2; ++s) {
            agx_index src = I->src[s];
            if (src.type != AGX_INDEX_IMMEDIATE &&
                src.type != AGX_INDEX_UNIFORM && modified &&
                (src.abs || src.neg || src.size == AGX_SIZE_16)) {
               agx_index plain = agx_temp(ctx, AGX_SIZE_32);
               if (src.size == AGX_SIZE_32 && src.abs && src.neg) {
                  /* -|x| has no grounded F32 move; clear then flip the
                   * sign bit (robustness2.*.vertex_attribute_fetch.
                   * no_fmt_qual.len_25{2,6}.*.1d.vert). */
                  agx_index bare = src;
                  bare.abs = bare.neg = false;
                  agx_index cleared = agx_temp(ctx, AGX_SIZE_32);
                  agx_bitop_to(&b, cleared, bare,
                               agx_mov_imm(&b, 32, 0x7fffffff), 0x8);
                  agx_bitop_to(&b, plain, cleared,
                               agx_mov_imm(&b, 32, 0x80000000), 0x6);
               } else {
                  agx_fmov_to(&b, plain, src);
               }
               I->src[s] = plain;
               continue;
            }
            /* An implicit zero is native, except beside a uniform in a tied
             * minimum/maximum: that form takes its other comparand from a
             * register. */
            const bool tied_uniform_minmax =
               I->op == AGX_OPCODE_FCMPSEL &&
               I->dest[0].size == AGX_SIZE_32 && !I->invert_cond &&
               (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
               I->src[s ^ 1].type == AGX_INDEX_UNIFORM &&
               agx_is_equiv(I->src[0], I->src[2]) &&
               agx_is_equiv(I->src[1], I->src[3]);
            if (src.type != AGX_INDEX_IMMEDIATE ||
                (src.value == 0 && !tied_uniform_minmax))
               continue;

            float value = agx_minifloat_decode(src.value);
            if (src.abs)
               value = fabsf(value);
            if (src.neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 32, fui(value));

            /* A tied minimum/maximum selects the same minifloat it compares.
             * The select slot reads a bit pattern, so keep the tie through
             * the register instead of leaving the minifloat there. */
            const agx_index sel = I->src[s + 2];
            if (I->op == AGX_OPCODE_FCMPSEL &&
                I->dest[0].size == AGX_SIZE_32 &&
                sel.type == AGX_INDEX_IMMEDIATE && sel.value == src.value &&
                sel.abs == src.abs && sel.neg == src.neg)
               I->src[s + 2] = I->src[s];
         }
      }

      /* Same-width F16 fmov is an FADD with negative zero. Apply sign
       * modifiers to the bits, preserving the half value and signed zero. */
      /* The 32-bit form has grounded |x| and -|x|+1 encodings only; -|x|
       * takes the same bit route (robustness2.*.vertex_attribute_fetch.
       * no_fmt_qual.len_25{2,6}.*.1d.vert). */
      /* The sign word belongs to the SOURCE's width. Materializing a half's
       * modifier through a 32-bit mask and a 32-bit temporary produced
       * `xor 4h, #0x80000000` into a 32-bit destination, which has no AGX3
       * encoding at all: every SPIR-V OpFNegate on a 16-bit value under a
       * float-controls preserve mode refused the whole shader
       * (dEQP-VK.spirv_assembly.instruction.*.float_controls.fp16.input_args.
       * negate_*_preserve*, 84 cases, while the fp32 rows passed).
       *
       * Absolute value and negation are also separate operations: |x| clears
       * the sign and -x flips it, so an unconditional XOR is only correct
       * when a negate is actually present. Apply each modifier that is set,
       * and leave an unmodified move alone. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FADD && !I->saturate &&
          (I->src[0].abs || I->src[0].neg) &&
          ((I->dest[0].size == AGX_SIZE_16 &&
            (I->src[0].size == AGX_SIZE_16 ||
             I->src[0].size == AGX_SIZE_32)) ||
           (I->dest[0].size == AGX_SIZE_32 && I->src[0].size == AGX_SIZE_32 &&
            I->src[0].abs && I->src[0].neg)) &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          ((I->src[1].value == 0 && I->src[1].neg) ||
           (I->src[1].value == agx_minifloat_encode(-0.0f) &&
            !I->src[1].neg && !I->src[1].abs))) {
         const enum agx_size size = I->src[0].size;
         const bool half = size == AGX_SIZE_16;
         const unsigned bits = half ? 16 : 32;
         const uint32_t sign_bit = half ? UINT32_C(0x8000) : UINT32_C(0x80000000);
         const uint32_t magnitude_mask =
            half ? UINT32_C(0x7fff) : UINT32_C(0x7fffffff);
         agx_index source = I->src[0];
         const bool absolute = source.abs, negate = source.neg;
         source.abs = false;
         source.neg = false;

         if (absolute) {
            agx_index mask = agx_mov_imm(&b, bits, magnitude_mask);
            agx_index result = agx_temp(ctx, size);
            agx_bitop_to(&b, result, source, mask, 0x8 /* AND */);
            source = result;
         }

         /* A half FADD against the negative-zero minifloat has a native
          * encoding only with the negate modifier still on its register
          * source (agx3_alu_hadd_neg_negzero). Once the modifier has been
          * applied to the bits the move is redundant anyway -- adding -0.0h
          * to an exact half is the identity -- so write the destination
          * directly and drop it. The 32-bit form keeps its move: that FADD
          * encodes, and removing it would change denormal and NaN handling
          * on a path that passes today. */
         const bool fold_into_dest = half &&
            I->dest[0].size == AGX_SIZE_16 && agx_channels(I->dest[0]) == 1 &&
            agx_channels(I->src[0]) == 1 && I->nr_dests == 1;

         if (negate) {
            agx_index sign = agx_mov_imm(&b, bits, sign_bit);
            agx_index result = (fold_into_dest && !absolute)
                                  ? I->dest[0]
                                  : agx_temp(ctx, size);
            agx_bitop_to(&b, result, source, sign, 0x6 /* XOR */);
            source = result;
         }

         if (fold_into_dest && negate && !absolute) {
            agx_remove_instruction(I);
            continue;
         }

         I->src[0] = source;
      }

      /* A fused control-flow comparison has no native nonzero-minifloat
       * endpoint on G17P. Chromium's three.js skinning fragment keeps such a
       * threshold around a texture-sampling branch, so it cannot be flattened
       * into an FSEL. Materialize the exact IEEE value while temporaries still
       * exist; the resulting IF/WHILE/BREAK_IF_FCMP uses the already-grounded
       * register/register compare. Keep zero inline for the distinct compact
       * zero-comparison form. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          (I->op == AGX_OPCODE_IF_FCMP ||
           I->op == AGX_OPCODE_WHILE_FCMP ||
           I->op == AGX_OPCODE_BREAK_IF_FCMP) &&
          I->nr_srcs == 2) {
         const bool immediate_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE;
         const unsigned immediate_src = immediate_first ? 0 : 1;
         const unsigned register_src = immediate_src ^ 1;

         if (I->src[immediate_src].type == AGX_INDEX_IMMEDIATE &&
             I->src[immediate_src].value != 0 &&
             I->src[register_src].type != AGX_INDEX_IMMEDIATE &&
             I->src[register_src].size == AGX_SIZE_32 &&
             !I->src[register_src].abs && !I->src[register_src].neg) {
            float value =
               agx_minifloat_decode(I->src[immediate_src].value);
            if (I->src[immediate_src].abs)
               value = fabsf(value);
            if (I->src[immediate_src].neg)
               value = -value;

            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            I->src[immediate_src] = agx_mov_imm(&b, 32, bits);
         }
      }

      /* G18G uses the G17G encoding before transcoding, so it does not run
       * the AGX3B-only control-flow source legalizer. Its native register
       * equality has no negate operand. Materialize floating-CF modifiers
       * with exact sign-bit operations before RA, keeping the comparison
       * and execution-mask semantics unchanged. */
      if (ctx->key->dev.emit_agx3d && I->nr_srcs == 2 &&
          (I->op == AGX_OPCODE_IF_FCMP ||
           I->op == AGX_OPCODE_WHILE_FCMP ||
           I->op == AGX_OPCODE_BREAK_IF_FCMP)) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].size != AGX_SIZE_32 ||
                agx_channels(I->src[s]) != 1 ||
                (!I->src[s].abs && !I->src[s].neg))
               continue;
            I->src[s] = agx3_materialize_f32_source(&b, I->src[s]);
         }
      }

      /* The G18G (G17G-family) full-register select compares f32 operands.
       * Retain a mediump comparator's value by widening it before that
       * comparison; leave the selected values and ordered comparison
       * condition intact. */
      if (agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 && I->nr_dests == 1 &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[2].size == AGX_SIZE_32 && I->src[3].size == AGX_SIZE_32) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                (I->src[s].size != AGX_SIZE_16 && I->src[s].size != AGX_SIZE_32) ||
                (I->src[s].size == AGX_SIZE_32 &&
                 !I->src[s].abs && !I->src[s].neg))
               continue;
            I->src[s] = agx3_materialize_f32_source(&b, I->src[s]);
         }
      }

      /* G18G's native FLTNORM consumes unmodified f32 sources. A negative
       * alpha from subtractive blending must reach its normal
       * clamp/quantization, not be dropped or have its sign modifier silently
       * ignored. */
      if (agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_FLTNORM_AGX3) {
         agx_foreach_src(I, s) {
            if (I->src[s].type != AGX_INDEX_IMMEDIATE &&
                !agx_is_null(I->src[s]) && I->src[s].size == AGX_SIZE_32 &&
                (I->src[s].abs || I->src[s].neg))
               I->src[s] = agx3_materialize_f32_source(&b, I->src[s]);
         }
      }

      /* Hyprland's colour-management shaders add a highp accumulator to a
       * mediump matrix term. There is no grounded plain mixed-width FADD on
       * G17P, so widen the half input explicitly and use the ordinary f32
       * register form. Keep the separately witnessed abs(f32) + f16 endpoint
       * available for its exact modifier shape. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FADD && I->nr_srcs == 2 && !I->saturate &&
          I->dest[0].size == AGX_SIZE_32) {
         for (unsigned s = 0; s < 2; ++s) {
            const unsigned other = s ^ 1;
            if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].size != AGX_SIZE_16 || I->src[s].abs ||
                I->src[s].neg || I->src[other].type == AGX_INDEX_IMMEDIATE ||
                I->src[other].size != AGX_SIZE_32 ||
                I->src[other].abs || I->src[other].neg)
               continue;

            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[s]);
            I->src[s] = widened;
         }
      }

      /* Native half FMA can carry a minifloat addend, but both multiplier
       * inputs are registers. Constant propagation commonly produces
       * clamp-like x * 0.5h + 0.5h; materialize only multiplier constants and
       * preserve the fused immediate-addend form. */
      if (I->op == AGX_OPCODE_HFMA && I->nr_srcs == 3) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type != AGX_INDEX_IMMEDIATE)
               continue;

            float value = agx_minifloat_decode(I->src[s].value);
            if (I->src[s].abs)
               value = fabsf(value);
            if (I->src[s].neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         }

         /* The immediate-addend form has no grounded multiplier-negate bit.
          * When either multiplicand is negated, load the addend as an exact
          * half so the general register FMA can carry the witnessed product
          * sign. This is refract's fused 1.0h - x*x operation. */
         if (I->src[2].type == AGX_INDEX_IMMEDIATE &&
             (I->src[0].neg || I->src[1].neg || I->src[2].abs ||
              I->src[2].neg)) {
            float value = agx_minifloat_decode(I->src[2].value);
            if (I->src[2].abs)
               value = fabsf(value);
            if (I->src[2].neg)
               value = -value;
            I->src[2] =
               agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         }

         /* The grounded half FMA carries negate but not absolute modifiers.
          * Clear a register's sign bit explicitly, preserving a following
          * negate as -abs(x), before register allocation. */
         for (unsigned s = 0; s < 3; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE || !I->src[s].abs)
               continue;

            const bool negate = I->src[s].neg;
            agx_index source = I->src[s];
            source.abs = false;
            source.neg = false;
            agx_index mask = agx_mov_imm(&b, 16, UINT16_C(0x7fff));
            agx_index absolute = agx_temp(ctx, AGX_SIZE_16);
            agx_bitop_to(&b, absolute, source, mask, 0x8 /* AND */);
            absolute.neg = negate;
            I->src[s] = absolute;
         }
      }

      /* Only saturation through -0.0h and 1.0h - x have grounded immediate
       * half-add forms. Materialize every other minifloat operand so general
       * mediump offsets use the exact register-register add encoding. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_HADD && I->nr_srcs == 2) {
         const bool saturate_negzero =
            I->saturate && I->src[1].type == AGX_INDEX_IMMEDIATE &&
            I->src[1].value == UINT8_C(0x80);
         const bool one_minus =
            !I->saturate && I->src[0].type != AGX_INDEX_IMMEDIATE &&
            I->src[0].neg && I->src[1].type == AGX_INDEX_IMMEDIATE &&
            I->src[1].value == UINT8_C(0x30);

         if (!saturate_negzero && !one_minus) {
            for (unsigned s = 0; s < 2; ++s) {
               if (I->src[s].type != AGX_INDEX_IMMEDIATE)
                  continue;

               float value = agx_minifloat_decode(I->src[s].value);
               if (I->src[s].abs)
                  value = fabsf(value);
               if (I->src[s].neg)
                  value = -value;
               I->src[s] =
                  agx_mov_imm(&b, 16, _mesa_float_to_half(value));
            }
         }
      }

      /* Half add has no register absolute field. Materialize abs by clearing
       * the sign bit, retaining a following negate for the FMA subtraction
       * lowering below. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_HADD && I->nr_srcs == 2) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE || !I->src[s].abs)
               continue;

            const bool negate = I->src[s].neg;
            agx_index source = I->src[s];
            source.abs = false;
            source.neg = false;
            agx_index mask = agx_mov_imm(&b, 16, UINT16_C(0x7fff));
            agx_index absolute = agx_temp(ctx, AGX_SIZE_16);
            agx_bitop_to(&b, absolute, source, mask, 0x8 /* AND */);
            absolute.neg = negate;
            I->src[s] = absolute;
         }
      }

      /* The recovered register-register half add has no grounded negate
       * field. Preserve signed sums through the witnessed FMA form:
       * a - b = fma(a, 1.0h, -b), while -a - b uses both grounded FMA sign
       * bits. Do this after materializing constants so terrain's 0.5h - x
       * shape reaches the same register FMA path. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_HADD && I->nr_srcs == 2 &&
          (I->src[0].neg || I->src[1].neg) && !I->src[0].abs &&
          !I->src[1].abs && I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].type != AGX_INDEX_IMMEDIATE) {
         if (I->src[0].neg && !I->src[1].neg) {
            const agx_index tmp = I->src[0];
            I->src[0] = I->src[1];
            I->src[1] = tmp;
         }

         const bool product_neg = I->src[0].neg;
         I->src[0].neg = false;
         I->op = AGX_OPCODE_HFMA;
         I->src[2] = I->src[1];
         I->src[1] =
            agx_mov_imm(&b, 16, _mesa_float_to_half(1.0f));
         I->src[1].neg = product_neg;
         I->nr_srcs = 3;
      }

      /* G17P's grounded scalar-half min/max takes four half registers. NIR
       * represents max(x, k) as a tied compare/select, so load one exact half
       * constant and reuse it in both tied positions. In particular, this is
       * the mediump max(x, 0.0h) generated by glmark2's bump fragment. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 && !I->invert_cond &&
          !I->saturate &&
          (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
          agx_is_equiv(I->src[0], I->src[2]) &&
          agx_is_equiv(I->src[1], I->src[3])) {
         const unsigned constant_pair =
            I->src[0].type == AGX_INDEX_IMMEDIATE ? 0 : 1;
         const unsigned register_pair = constant_pair ^ 1;

         if (I->src[constant_pair].type == AGX_INDEX_IMMEDIATE &&
             I->src[register_pair].type != AGX_INDEX_IMMEDIATE &&
             I->src[register_pair].size == AGX_SIZE_16 &&
             !I->src[constant_pair].abs && !I->src[constant_pair].neg &&
             !I->src[register_pair].abs && !I->src[register_pair].neg) {
            const float value =
               agx_minifloat_decode(I->src[constant_pair].value);
            const agx_index materialized =
               agx_mov_imm(&b, 16, _mesa_float_to_half(value));
            I->src[constant_pair] = materialized;
            I->src[constant_pair + 2] = materialized;
         }
      }

      /* The grounded mixed register/immediate FSEL can select an f32 register
       * or minifloat after comparing two f32 registers. Normalize a narrow
       * first comparator to full precision rather than guessing an unwitnessed
       * reversed-immediate or mixed-width encoding. The constant can already
       * be a compact half register or an inline minifloat; the latter is the
       * point-size clamp generated by glmark2's ideas vertex shader. A zero
       * first comparator stays inline: the register/zero FSEL encodes it by
       * exchanging the comparators, as Apple does for max(0.0, x). */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 &&
          ((I->src[0].type == AGX_INDEX_IMMEDIATE && I->src[0].value != 0) ||
           (I->src[0].type != AGX_INDEX_IMMEDIATE &&
            I->src[0].size == AGX_SIZE_16 && !I->src[0].abs &&
            !I->src[0].neg)) &&
          I->src[1].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].size == AGX_SIZE_32 &&
          !((I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
            !I->invert_cond && !I->saturate &&
            agx_is_equiv(I->src[0], I->src[2]) &&
            agx_is_equiv(I->src[1], I->src[3])) &&
          ((I->src[2].type == AGX_INDEX_IMMEDIATE &&
            I->src[3].type != AGX_INDEX_IMMEDIATE &&
            I->src[3].size == AGX_SIZE_32) ||
           (I->src[3].type == AGX_INDEX_IMMEDIATE &&
            I->src[2].type != AGX_INDEX_IMMEDIATE &&
            I->src[2].size == AGX_SIZE_32))) {
         if (I->src[0].type == AGX_INDEX_IMMEDIATE) {
            float value = agx_minifloat_decode(I->src[0].value);
            if (I->src[0].abs)
               value = fabsf(value);
            if (I->src[0].neg)
               value = -value;

            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            I->src[0] = agx_mov_imm(&b, 32, bits);
         } else {
            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[0]);
            I->src[0] = widened;
         }
      }

      /* Motion-blur loop bounds compare a highp value against a minifloat and
       * select a half-register boolean or a raw 16-bit immediate. Neither
       * mixed axis is grounded together, so materialize both constants and
       * use the general four-register half FSEL. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 && !I->saturate) {
         const bool compare_immediate_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE;
         const unsigned compare_immediate = compare_immediate_first ? 0 : 1;
         const unsigned compare_register = compare_immediate ^ 1;
         const bool select_immediate_first =
            I->src[2].type == AGX_INDEX_IMMEDIATE;
         const unsigned select_immediate = select_immediate_first ? 2 : 3;
         const unsigned select_register = select_immediate == 2 ? 3 : 2;

         if (I->src[compare_immediate].type == AGX_INDEX_IMMEDIATE &&
             I->src[compare_register].type != AGX_INDEX_IMMEDIATE &&
             I->src[compare_register].size == AGX_SIZE_32 &&
             !I->src[compare_register].abs &&
             !I->src[compare_register].neg &&
             I->src[select_immediate].type == AGX_INDEX_IMMEDIATE &&
             I->src[select_immediate].value <= 1 &&
             !I->src[select_immediate].abs &&
             !I->src[select_immediate].neg &&
             I->src[select_register].type != AGX_INDEX_IMMEDIATE &&
             I->src[select_register].size == AGX_SIZE_16 &&
             !I->src[select_register].abs &&
             !I->src[select_register].neg) {
            float value =
               agx_minifloat_decode(I->src[compare_immediate].value);
            if (I->src[compare_immediate].abs)
               value = fabsf(value);
            if (I->src[compare_immediate].neg)
               value = -value;
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            I->src[compare_immediate] = agx_mov_imm(&b, 32, bits);
            I->src[select_immediate] =
               agx_mov_imm(&b, 16, I->src[select_immediate].value);
         }
      }

      /* A mediump zero test selecting either a highp value or zero can use
       * the grounded full-register/zero mixed-result FSEL once its comparator
       * is widened. Leave both zero immediates inline so existing compact
       * encodings, including stock vkcube, remain byte-identical. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 && !I->saturate &&
          ((I->src[0].type == AGX_INDEX_IMMEDIATE &&
            I->src[0].value == 0 && I->src[1].type != AGX_INDEX_IMMEDIATE &&
            I->src[1].size == AGX_SIZE_16) ||
           (I->src[1].type == AGX_INDEX_IMMEDIATE &&
            I->src[1].value == 0 && I->src[0].type != AGX_INDEX_IMMEDIATE &&
            I->src[0].size == AGX_SIZE_16)) &&
          ((I->src[2].type == AGX_INDEX_IMMEDIATE &&
            I->src[2].value == 0 && I->src[3].type != AGX_INDEX_IMMEDIATE &&
            I->src[3].size == AGX_SIZE_32) ||
           (I->src[3].type == AGX_INDEX_IMMEDIATE &&
            I->src[3].value == 0 && I->src[2].type != AGX_INDEX_IMMEDIATE &&
            I->src[2].size == AGX_SIZE_32))) {
         const unsigned half_src =
            I->src[0].type == AGX_INDEX_IMMEDIATE ? 1 : 0;
         /* A uniform half is compared in place (agx3_alu_fcmp_select's
          * uniform comparand), as Apple does. */
         if (!I->src[half_src].abs && !I->src[half_src].neg &&
             I->src[half_src].type != AGX_INDEX_UNIFORM) {
            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[half_src]);
            I->src[half_src] = widened;
         }
      }

      /* Hyprland colour management compares mediump values against compact
       * constants and compares highp values against abs(highp), selecting two
       * independent highp registers. Normalize those comparator-only shapes
       * to the general four-register f32 select. This avoids extending a
       * modifier or mixed-width field beyond the endpoints witnessed for it. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 && !I->saturate &&
          I->fcond == AGX_FCOND_GE &&
          ((I->src[0].type != AGX_INDEX_IMMEDIATE &&
            I->src[0].size == AGX_SIZE_16) ||
           (I->src[1].type != AGX_INDEX_IMMEDIATE &&
            I->src[1].size == AGX_SIZE_16) ||
           I->src[0].abs || I->src[1].abs) &&
          ((I->src[2].type == AGX_INDEX_IMMEDIATE && !I->src[2].abs &&
            !I->src[2].neg) ||
           (I->src[2].size == AGX_SIZE_32 && !I->src[2].abs &&
            !I->src[2].neg)) &&
          ((I->src[3].type == AGX_INDEX_IMMEDIATE && !I->src[3].abs &&
            !I->src[3].neg) ||
           (I->src[3].size == AGX_SIZE_32 && !I->src[3].abs &&
            !I->src[3].neg)) &&
          (I->src[2].type != AGX_INDEX_IMMEDIATE ||
           I->src[3].type != AGX_INDEX_IMMEDIATE)) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE) {
               float value = agx_minifloat_decode(I->src[s].value);
               if (I->src[s].abs)
                  value = fabsf(value);
               if (I->src[s].neg)
                  value = -value;

               uint32_t bits;
               memcpy(&bits, &value, sizeof(bits));
               I->src[s] = agx_mov_imm(&b, 32, bits);
            } else if (I->src[s].size == AGX_SIZE_16 &&
                       !I->src[s].abs && !I->src[s].neg) {
               agx_index widened = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, widened, I->src[s]);
               I->src[s] = widened;
            } else if (I->src[s].size == AGX_SIZE_32 && I->src[s].abs &&
                       !I->src[s].neg) {
               agx_index source = I->src[s];
               source.abs = false;
               agx_index mask =
                  agx_mov_imm(&b, 32, UINT32_C(0x7fffffff));
               agx_index absolute = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, absolute, source, mask, 0x8 /* AND */);
               I->src[s] = absolute;
            }
         }
      }

      /* Conditional discard lowers a standalone comparison to byte selection
       * after register allocation. Materialize a nonzero minifloat comparator
       * while a temporary can still be allocated, so Hyprland's alpha == 1.0
       * predicate reaches the grounded two-register comparison form. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMP && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_16 &&
          ctx->stage == MESA_SHADER_FRAGMENT &&
          ctx->nir->info.fs.uses_discard) {
         const bool immediate_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE;
         const unsigned immediate_src = immediate_first ? 0 : 1;
         const unsigned register_src = immediate_src ^ 1;

         if (I->src[immediate_src].type == AGX_INDEX_IMMEDIATE &&
             I->src[immediate_src].value != 0 &&
             I->src[register_src].type != AGX_INDEX_IMMEDIATE &&
             I->src[register_src].size == AGX_SIZE_32 &&
             !I->src[register_src].abs && !I->src[register_src].neg) {
            float value =
               agx_minifloat_decode(I->src[immediate_src].value);
            if (I->src[immediate_src].abs)
               value = fabsf(value);
            if (I->src[immediate_src].neg)
               value = -value;

            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            I->src[immediate_src] = agx_mov_imm(&b, 32, bits);
         }
      }

      /* Standalone half comparisons lower after RA to a byte-constant FSEL.
       * Normalize their zero comparison here, while a temporary can still be
       * allocated, so late pseudo lowering reaches the same grounded form. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMP && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_16 &&
          (I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT ||
           I->fcond == AGX_FCOND_GT || I->fcond == AGX_FCOND_GE ||
           I->fcond == AGX_FCOND_LE)) {
         const bool zero_first = I->src[0].type == AGX_INDEX_IMMEDIATE &&
                                 I->src[0].value == 0;
         const unsigned value_src = zero_first ? 1 : 0;
         const unsigned zero_src = zero_first ? 0 : 1;

         if (I->src[zero_src].type == AGX_INDEX_IMMEDIATE &&
             I->src[zero_src].value == 0 &&
             I->src[value_src].type != AGX_INDEX_IMMEDIATE &&
             I->src[value_src].size == AGX_SIZE_16 &&
             !I->src[value_src].abs && !I->src[value_src].neg) {
            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[value_src]);
            I->src[value_src] = widened;
         }
      }

      /* The general half compare/select takes four registers. Selected
       * immediates are integer bit patterns, including discard's 0/1 masks.
       * Tied min/max constants were materialized above. Keep f32 comparisons
       * selecting byte constants on their separate grounded form. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[0].size == AGX_SIZE_16 &&
          I->src[1].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].size == AGX_SIZE_16 &&
          (I->src[2].type == AGX_INDEX_IMMEDIATE ||
           I->src[3].type == AGX_INDEX_IMMEDIATE) &&
          (I->src[2].type == AGX_INDEX_IMMEDIATE ||
           I->src[2].size == AGX_SIZE_16) &&
          (I->src[3].type == AGX_INDEX_IMMEDIATE ||
           I->src[3].size == AGX_SIZE_16)) {
         for (unsigned s = 2; s < 4; ++s) {
            if (I->src[s].type != AGX_INDEX_IMMEDIATE)
               continue;

            if (ctx->key->dev.chip == AGX_CHIP_G17P) {
               I->src[s] = agx_mov_imm(&b, 16, I->src[s].value);
               continue;
            }

            float value = agx_minifloat_decode(I->src[s].value);
            if (I->src[s].abs)
               value = fabsf(value);
            if (I->src[s].neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         }
      }

      /* A half comparison against a minifloat has grounded G17P FSEL forms
       * once the register comparator is full-width. Widen the half operand
       * explicitly for a half result selected from registers and/or
       * minifloats instead of treating it as a nonexistent mixed-width
       * immediate form. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 &&
          (I->src[2].type == AGX_INDEX_IMMEDIATE ||
           I->src[3].type == AGX_INDEX_IMMEDIATE) &&
          (I->src[2].type == AGX_INDEX_IMMEDIATE ||
           I->src[2].size == AGX_SIZE_16) &&
          (I->src[3].type == AGX_INDEX_IMMEDIATE ||
           I->src[3].size == AGX_SIZE_16)) {
         const bool immediate_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE;
         const unsigned value_src = immediate_first ? 1 : 0;
         const unsigned immediate_src = immediate_first ? 0 : 1;

         if (I->src[immediate_src].type == AGX_INDEX_IMMEDIATE &&
             I->src[value_src].type != AGX_INDEX_IMMEDIATE &&
             I->src[value_src].size == AGX_SIZE_16 &&
             !I->src[value_src].abs && !I->src[value_src].neg) {
            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[value_src]);
            I->src[value_src] = widened;
         }
      }

      /* A half comparison against a minifloat selecting two half registers
       * has no separately grounded immediate form. Materialize the compare
       * constant so it uses the general four-register half select whose
       * conditions and lifetime fields are already recovered. */
      if (agx_isa_is_agx3(ctx->key->dev.encoding) &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 &&
          I->src[2].type != AGX_INDEX_IMMEDIATE &&
          I->src[2].size == AGX_SIZE_16 &&
          I->src[3].type != AGX_INDEX_IMMEDIATE &&
          I->src[3].size == AGX_SIZE_16) {
         for (unsigned s = 0; s < 2; ++s) {
            const unsigned other = s ^ 1;
            if (I->src[s].type != AGX_INDEX_IMMEDIATE ||
                I->src[other].type == AGX_INDEX_IMMEDIATE ||
                I->src[other].size != AGX_SIZE_16)
               continue;

            float value = agx_minifloat_decode(I->src[s].value);
            if (I->src[s].abs)
               value = fabsf(value);
            if (I->src[s].neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         }
      }

      /* Multiplication is commutative.  Keep a sole immediate in the second
       * multiply slot, matching Apple's native IMAD form and the existing
       * grounded G17P encoder.  Geometry/XFB arithmetic naturally arrives
       * from NIR with the constant first. */
      if (I->op == AGX_OPCODE_IMAD && I->nr_srcs == 3 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].type != AGX_INDEX_IMMEDIATE) {
         agx_index tmp = I->src[0];
         I->src[0] = I->src[1];
         I->src[1] = tmp;
      }

      /* G17P has a register half multiply, but its immediate half form
       * requires an addend. Keep a pure half multiply in the witnessed
       * register domain instead of passing an unencodable immediate to the
       * packer. The low-half product has the same modulo-2^16 value. */
      if (ctx->key->dev.chip == AGX_CHIP_G17P &&
          I->op == AGX_OPCODE_IMAD && I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_16 && I->shift == 0 && !I->saturate &&
          agx_channels(I->dest[0]) == 1 &&
          I->src[0].size == AGX_SIZE_16 && agx_channels(I->src[0]) == 1 &&
          !I->src[0].abs && !I->src[0].neg &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].value <= UINT8_MAX &&
          !I->src[1].abs && !I->src[1].neg &&
          I->src[2].type == AGX_INDEX_IMMEDIATE &&
          I->src[2].value == 0 && !I->src[2].abs && !I->src[2].neg)
         I->src[1] = agx_mov_imm(&b, 16, I->src[1].value);

      /* The established 32x32->64 multiply uses register multiplicands.
       * An unsigned byte immediate has the same value as a zero-extended
       * dword register. Materialize it before RA instead of asking the
       * packer for an unsupported wide immediate form. Keep its zero case
       * available for the existing multiply-by-zero fold. G18G only. */
      if (agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_IMAD && I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_64 && I->shift == 0 &&
          I->src[1].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].value > 0 && I->src[1].value <= UINT8_MAX &&
          !I->src[1].abs && !I->src[1].neg) {
         I->src[1] = agx_abs(agx_mov_imm(&b, 32, I->src[1].value));
      }

      /* Neither register nor immediate multiplication has a grounded
       * negative immediate addend. Split a * b - byte into the native
       * pure multiply followed by subtract-immediate, with the same
       * modulo-2^32 result. */
      if (I->op == AGX_OPCODE_IMAD && I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_32 && I->shift == 0 &&
          !I->src[1].abs && !I->src[1].neg &&
          ((I->src[1].type == AGX_INDEX_IMMEDIATE &&
            I->src[1].value <= UINT8_MAX) ||
           (I->src[1].type == AGX_INDEX_NORMAL &&
            I->src[1].size == AGX_SIZE_32)) &&
          I->src[2].type == AGX_INDEX_IMMEDIATE && !I->src[2].abs &&
          I->src[2].neg && I->src[2].value <= UINT8_MAX) {
         agx_index product = agx_temp(ctx, AGX_SIZE_32);
         agx_imad_to(&b, product, I->src[0], I->src[1], agx_zero(), 0);
         I->op = AGX_OPCODE_IADD;
         I->src[0] = product;
         I->src[1] = I->src[2];
         I->nr_srcs = 2;
      }

      if ((I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL) &&
          I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[2].type == AGX_INDEX_IMMEDIATE) {
         const bool immediate_shift =
            (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
             ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
            I->bfi_mask == 0 && I->src[0].type == AGX_INDEX_IMMEDIATE &&
            I->src[0].value == 0 && I->src[1].size == AGX_SIZE_32 &&
            I->src[2].value >= 1 && I->src[2].value <= 31 &&
            !I->src[2].abs && !I->src[2].neg;
         /* A constant-offset, constant-width extract
          * is one SRL with the offset in f30 and the width in f57, as Apple
          * emits for (v >> 8) & 63 (native fragment 0x118,
          * a700540203002200f0106100: f30=8 f57=6). agx3_alu_bfeil() already
          * encodes that immediate form; only this materialization of the
          * offset into a register (an extra LDIMM) stood in the way. */
         /* M5 has the same form (agx3_alu_bfeil_m5): its SRL takes the
          * offset as an immediate with or without a width. G18G keeps the
          * register offset. */
         const bool immediate_extract =
            (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
             (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
              !ctx->key->dev.emit_agx3d)) &&
            I->op == AGX_OPCODE_BFEIL && I->bfi_mask >= 1 &&
            I->bfi_mask <= 31 && I->src[0].type == AGX_INDEX_IMMEDIATE &&
            I->src[0].value == 0 && I->src[1].size == AGX_SIZE_32 &&
            I->src[2].value <= 31 && !I->src[2].abs && !I->src[2].neg;
         if (!immediate_shift && !immediate_extract)
            I->src[2] = agx_mov_imm(&b, 32, I->src[2].value);
      }

      /* G17P's register-selected shift has no immediate value operand.  NIR
       * uses BFI(0, 1, shift) for 1u << shift, so materialize that value just
       * like the variable shift amount above. */
      if ((I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL) &&
          I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[1].type == AGX_INDEX_IMMEDIATE) {
         I->src[1] = agx_mov_imm(&b, 32, I->src[1].value);
      } else if (I->op == AGX_OPCODE_BITOP && I->nr_dests == 1 &&
                 (I->dest[0].size == AGX_SIZE_32 ||
                  (I->dest[0].size == AGX_SIZE_16 &&
                   (I->truth_table == 0x8 || I->truth_table == 0xe ||
                    I->truth_table == 0x6 ||
                    (agx_is_g18g(&ctx->key->dev) &&
                     (I->truth_table == 0x1 || I->truth_table == 0x2 ||
                      I->truth_table == 0x4 || I->truth_table == 0x7 ||
                      I->truth_table == 0x9 || I->truth_table == 0xb ||
                      I->truth_table == 0xd)))))) {
         /* G18G's seven complemented halfword tables are grounded for
          * register operands. Sample-mask lowering retains constants in
          * XNOR and ANDC; materialize them before RA without changing the
          * truth table or the operand order (including swapped ANDC/ORC). */
         /* The packer encodes AND/OR/XOR against an 8-bit immediate directly
          * (agx3_alu_bit_imm: f37=1 selects the immediate operand kind, f29
          * holds the constant; witnessed Apple bytes in agx3_alu_core.h, e.g.
          * `a & 0xff` 0b011fff100200800000). Native G17P code uses that form
          * (native fragment 0x124 `bit_and ... f29=63`). Materializing the
          * constant costs an extra ldimm per use. Keep the immediate only for
          * the exact shape that encoder accepts. */
         if (agx_agx3_bitop_immediate_ok(ctx, I))
            continue;
         agx_foreach_src(I, s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE)
               I->src[s] = agx_mov_imm(&b, I->dest[0].size == AGX_SIZE_16 ? 16 : 32,
                                      I->src[s].value);
         }
      }
   }
}

void
agx_lower_agx3_immediates(agx_context *ctx)
{
   if (!agx_is_g18g(&ctx->key->dev))
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_MOV_IMM || I->nr_dests != 1 ||
          I->dest[0].size != AGX_SIZE_16 || agx_channels(I->dest[0]) != 1)
         continue;

      /* G18G (G17G family) has an established dword immediate load, but its
       * half load encoding has not been established. Materialize the raw half bits
       * in a dword and select the low half before register allocation. This
       * is a bit-preserving operation, including for signed zero and NaNs;
       * it is not a floating-point conversion. Run after ALU immediate
       * materialization, which can introduce more half constants.
       */
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index word = agx_mov_imm(&b, 32, I->imm & UINT16_MAX);
      agx_instr *split = agx_split(&b, 2, word);
      split->dest[0] = I->dest[0];
      split->dest[1] = agx_temp(ctx, AGX_SIZE_16);
      agx_remove_instruction(I);
   }
}

void
agx_materialize_agx3_cf_constants(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if ((I->op != AGX_OPCODE_IF_FCMP &&
           I->op != AGX_OPCODE_WHILE_FCMP &&
           I->op != AGX_OPCODE_BREAK_IF_FCMP) ||
          I->nr_srcs != 2)
         continue;

      /* A half operand compared against a float one reaches here once the
       * optimizer has propagated the widening into the comparison. Apple's
       * G17P compiler has no mixed-width control-flow compare: for
       * `if (f < h)` it widens h with an explicit conversion first, then
       * compares the f32 pair. Do the same.
       */
      for (unsigned s = 0; s < 2; ++s) {
         if (I->src[s].type != AGX_INDEX_NORMAL ||
             I->src[s].size != AGX_SIZE_16 ||
             I->src[s ^ 1].size != AGX_SIZE_32)
            continue;

         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide, I->src[s]);
         I->src[s] = wide;
      }

      /* The long native ABS compare has no grounded unordered selector.
       * Ordered GE/LE are represented as an inverted unordered LT/GT, so an
       * ABS modifier on either input cannot use that encoding point.  Keep
       * the direct ABS form for EQ/LT/GT, and materialize only the otherwise
       * unencodable modifier combination as exact IEEE sign-bit operations
       * before register allocation.  This is also the safe path when ABS and
       * NEG occur together, since the native modifier forms are disjoint. */
      const bool unordered_relation =
         I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN ||
         I->fcond == AGX_FCOND_GE || I->fcond == AGX_FCOND_LE;
      const bool has_abs = I->src[0].abs || I->src[1].abs;
      const bool has_neg = I->src[0].neg || I->src[1].neg;
      /* Register/register negate is grounded only for strict LT/GT with
       * one negated source. Keep that form and the independently supported
       * register/zero comparisons; materialize signs for the remaining
       * equality, unordered and doubly-negated register predicates. */
      const bool plain_register_pair =
         I->src[0].type == AGX_INDEX_NORMAL &&
         I->src[1].type == AGX_INDEX_NORMAL &&
         I->src[0].size == AGX_SIZE_32 && I->src[1].size == AGX_SIZE_32;
      const bool native_neg =
         (I->fcond == AGX_FCOND_LT || I->fcond == AGX_FCOND_GT) &&
         (I->src[0].neg != I->src[1].neg);
      if ((has_abs && (unordered_relation || has_neg)) ||
          (plain_register_pair && has_neg && !native_neg)) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type != AGX_INDEX_NORMAL ||
                I->src[s].size != AGX_SIZE_32 ||
                (!I->src[s].abs && !I->src[s].neg))
               continue;

            const bool absolute = I->src[s].abs;
            const bool negate = I->src[s].neg;
            agx_index value = I->src[s];
            value.abs = false;
            value.neg = false;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

            if (absolute) {
               agx_index mask =
                  agx_mov_imm(&b, 32, UINT32_C(0x7fffffff));
               agx_index result = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, result, value, mask, 0x8 /* AND */);
               value = result;
            }

            if (negate) {
               agx_index sign =
                  agx_mov_imm(&b, 32, UINT32_C(0x80000000));
               agx_index result = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, result, value, sign, 0x6 /* XOR */);
               value = result;
            }

            I->src[s] = value;
         }
      }

      /* G17P's structured float-CF path is grounded for f32 register pairs
       * and for literal zero. Firefox's WebRender fragment compares a live
       * f32 against 1.0 (and later -10.0), which reaches AIR as a nonzero
       * minifloat. Materialize only that missing register/minifloat axis into
       * an exact f32 LDIMM before RA; this adds no speculative CF encoding and
       * leaves the shorter native zero form byte-identical. */
      for (unsigned s = 0; s < 2; ++s) {
         const unsigned other = s ^ 1;
         if (I->src[s].type != AGX_INDEX_IMMEDIATE ||
             I->src[s].value == 0 ||
             I->src[other].type == AGX_INDEX_IMMEDIATE ||
             I->src[other].size != AGX_SIZE_32 || I->src[other].memory ||
             I->src[other].abs || I->src[other].neg)
            continue;

         float value = agx_minifloat_decode(I->src[s].value);
         if (I->src[s].abs)
            value = fabsf(value);
         if (I->src[s].neg)
            value = -value;

         uint32_t bits;
         memcpy(&bits, &value, sizeof(bits));
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[s] = agx_mov_imm(&b, 32, bits);
      }
   }
}

/* Half-precision ALU instructions cannot encode 32-bit register sources.
 * Convert them explicitly. Immediate and null sources have no size bit. */
static agx_index
agx_coerce_f16_src(agx_builder *b, agx_index src)
{
   if (src.size == AGX_SIZE_16 || src.type == AGX_INDEX_IMMEDIATE ||
       src.type == AGX_INDEX_NULL)
      return src;

   agx_index tmp = agx_temp(b->shader, AGX_SIZE_16);
   agx_fmov_to(b, tmp, src);
   return tmp;
}

/* Ordinary G17P f32 arithmetic has no generally witnessed scalar-half
 * register source.  Widen such operands explicitly with the native
 * half-to-float add form before register allocation.  Immediate and null
 * sources have no size bit and remain directly encodable. */
static agx_index
agx_g17p_coerce_f32_src(agx_builder *b, agx_index src)
{
   if (!agx_isa_is_agx3(b->shader->key->dev.encoding) ||
       src.size == AGX_SIZE_32 || src.type == AGX_INDEX_IMMEDIATE ||
       src.type == AGX_INDEX_NULL)
      return src;

   assert(src.size == AGX_SIZE_16);
   agx_index tmp = agx_temp(b->shader, AGX_SIZE_32);
   agx_fmov_to(b, tmp, src);
   return tmp;
}

static agx_index
agx_g17p_shift_boundary(agx_builder *b, unsigned shift)
{
   assert(shift >= 1 && shift <= 31);
   agx_index positive = agx_mov_imm(b, 32, shift);
   agx_index boundary = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, boundary, agx_zero(), positive, 0);
   return boundary;
}

static agx_index
agx_g17p_ushr_imm(agx_builder *b, agx_index value, unsigned shift)
{
   return agx_ushr(b, value, agx_g17p_shift_boundary(b, shift));
}

static agx_index
agx_g17p_and_imm(agx_builder *b, agx_index value, uint32_t mask)
{
   return agx_and(b, value, agx_mov_imm(b, 32, mask));
}

static agx_index
agx_g17p_ishl_imm(agx_builder *b, agx_index value, unsigned shift)
{
   agx_index shifted = agx_temp(b->shader, AGX_SIZE_32);
   agx_bfi_to(b, shifted, agx_immediate(0), value,
              agx_immediate(shift), 0);
   return shifted;
}

/* nir_op_interleave is the exact 16+16 -> 32 Morton permutation.
 * No G17P/G18G INTL encoding is established. Expand before RA using the same
 * grounded shifts and bitwise operations as the bit-reversal lowering.
 */
static agx_index
agx_g17p_dilate_u16(agx_builder *b, agx_index value)
{
   agx_index x = agx_temp(b->shader, AGX_SIZE_32);
   agx_mov_to(b, x, value);
   x = agx_g17p_and_imm(b, x, UINT32_C(0xffff));
   static const struct {
      uint8_t shift;
      uint32_t mask;
   } stages[] = {
      {8, UINT32_C(0x00ff00ff)},
      {4, UINT32_C(0x0f0f0f0f)},
      {2, UINT32_C(0x33333333)},
      {1, UINT32_C(0x55555555)},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
      x = agx_g17p_and_imm(
         b, agx_or(b, x, agx_g17p_ishl_imm(b, x, stages[i].shift)),
         stages[i].mask);
   }
   return x;
}

static agx_instr *
agx_emit_g17p_interleave(agx_builder *b, agx_index dst,
                         agx_index even, agx_index odd)
{
   assert(dst.size == AGX_SIZE_32);
   assert(even.size == AGX_SIZE_16 && odd.size == AGX_SIZE_16);
   agx_index x = agx_g17p_dilate_u16(b, even);
   agx_index y = agx_g17p_dilate_u16(b, odd);
   return agx_or_to(b, dst, x, agx_g17p_ishl_imm(b, y, 1));
}

static agx_instr *
agx_emit_g17p_bitrev(agx_builder *b, agx_index dst, agx_index value)
{
   /* AGX3's BITREV opcode is not grounded. Express GLSL
    * bitfieldReverse() with the ordinary G17P shifts and bitwise operations
    * instead. Every shift below is a compile-time constant and every mask is
    * exact, so this is the conventional five-stage 32-bit permutation rather
    * than an approximation of an unknown native instruction. */
   static const struct {
      uint8_t shift;
      uint32_t mask;
   } stages[] = {
      {1, UINT32_C(0x55555555)},
      {2, UINT32_C(0x33333333)},
      {4, UINT32_C(0x0f0f0f0f)},
      {8, UINT32_C(0x00ff00ff)},
   };

   agx_index x = value;
   for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
      agx_index lo = agx_g17p_and_imm(b, x, stages[i].mask);
      lo = agx_g17p_ishl_imm(b, lo, stages[i].shift);
      agx_index hi = agx_g17p_and_imm(
         b, agx_g17p_ushr_imm(b, x, stages[i].shift), stages[i].mask);
      x = agx_or(b, lo, hi);
   }

   return agx_or_to(b, dst, agx_g17p_ushr_imm(b, x, 16),
                    agx_g17p_ishl_imm(b, x, 16));
}

static agx_instr *
agx_emit_g17p_ufind_msb(agx_builder *b, agx_index dst, agx_index value)
{
   /* FT1 on G17P exposes a non-architectural logarithmic intermediate. Find
    * the top bit with the integer forms already qualified on hardware:
    * smear the top bit downward, popcount the resulting prefix with SWAR,
    * then subtract one. Zero smears/counts to zero and therefore returns -1
    * without a compare or select. */
   assert(value.size == AGX_SIZE_32 && dst.size == AGX_SIZE_32);

   agx_index scan = value;
   static const unsigned smear_shifts[] = {1, 2, 4, 8, 16};
   for (unsigned i = 0; i < ARRAY_SIZE(smear_shifts); ++i)
      scan = agx_or(b, scan, agx_g17p_ushr_imm(b, scan, smear_shifts[i]));

   agx_index odd = agx_g17p_and_imm(
      b, agx_g17p_ushr_imm(b, scan, 1), UINT32_C(0x55555555));
   scan = agx_iadd(b, scan, agx_neg(odd), 0);

   agx_index pairs_lo = agx_g17p_and_imm(b, scan, UINT32_C(0x33333333));
   agx_index pairs_hi = agx_g17p_and_imm(
      b, agx_g17p_ushr_imm(b, scan, 2), UINT32_C(0x33333333));
   scan = agx_iadd(b, pairs_lo, pairs_hi, 0);

   scan = agx_g17p_and_imm(
      b, agx_iadd(b, scan, agx_g17p_ushr_imm(b, scan, 4), 0),
      UINT32_C(0x0f0f0f0f));
   scan = agx_iadd(b, scan, agx_g17p_ushr_imm(b, scan, 8), 0);
   scan = agx_iadd(b, scan, agx_g17p_ushr_imm(b, scan, 16), 0);

   agx_index count = agx_g17p_and_imm(b, scan, 0x3f);
   return agx_iadd_to(b, dst, count, agx_neg(agx_immediate(1)), 0);
}

static agx_instr *
agx_emit_alu(agx_builder *b, nir_alu_instr *instr)
{
   unsigned srcs = nir_op_infos[instr->op].num_inputs;
   unsigned sz = instr->def.bit_size;
   unsigned src_sz = srcs ? nir_src_bit_size(instr->src[0].src) : 0;
   ASSERTED unsigned comps = instr->def.num_components;

   assert(comps == 1 || nir_op_is_vec_or_mov(instr->op));
   assert(sz == 1 ||
          ((nir_op_is_vec_or_mov(instr->op) ||
            is_conversion_to_8bit(instr->op) || instr->op == nir_op_bcsel) &&
           sz == 8) ||
          sz == 16 || sz == 32 || sz == 64);

   agx_index dst = agx_def_index(&instr->def);
   agx_index s0 = srcs > 0 ? agx_alu_src_index(b, instr->src[0]) : agx_null();
   agx_index s1 = srcs > 1 ? agx_alu_src_index(b, instr->src[1]) : agx_null();
   agx_index s2 = srcs > 2 ? agx_alu_src_index(b, instr->src[2]) : agx_null();
   agx_index s3 = srcs > 3 ? agx_alu_src_index(b, instr->src[3]) : agx_null();

   agx_index i0 = agx_immediate(0);
   agx_index i1 = agx_immediate(1);

#define UNOP(nop, aop)                                                         \
   case nir_op_##nop:                                                          \
      return agx_##aop##_to(b, dst, s0);
#define BINOP(nop, aop)                                                        \
   case nir_op_##nop:                                                          \
      return agx_##aop##_to(b, dst, s0, s1);

   switch (instr->op) {
      UNOP(f2f16, fmov);
      UNOP(f2f16_rtne, fmov);
      UNOP(f2f32, fmov);
      UNOP(fround_even, roundeven);
      UNOP(ftrunc, trunc);
      UNOP(ffloor, floor);
      UNOP(fceil, ceil);
      UNOP(frcp, rcp);
      UNOP(frsq, rsqrt);
      UNOP(flog2, log2);
      UNOP(fsinc_agx3, sinc_agx3);
      UNOP(fexp2, exp2);

      UNOP(mov, mov);
   case nir_op_u2u32:
      /* A pixel coordinate is at most 16 bits and
       * its native special register is already zero-extended to 32 bits, so
       * reuse that register instead of copying the half and clearing the
       * high half (two instructions per coordinate). */
      if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
          s0.size == AGX_SIZE_16 && dst.size == AGX_SIZE_32) {
         for (unsigned i = 0; i < 2; ++i) {
            if (!agx_is_null(b->shader->agx3_pixel_coord_native[i]) &&
                agx_is_equiv(s0, b->shader->agx3_pixel_coord_half[i]))
               return agx_mov_to(b, dst,
                                 b->shader->agx3_pixel_coord_native[i]);
         }
      }
      return agx_mov_to(b, dst, s0);
   case nir_op_bitfield_reverse:
      /* AGX3B only: G17P's emulated sequence uses an immediate+immediate IADD
       * that AGX3C cannot encode; AGX3C keeps its native BITREV. */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          src_sz == 32)
         return agx_emit_g17p_bitrev(b, dst, s0);
      return agx_bitrev_to(b, dst, s0);
   case nir_op_bit_count:
      if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
          src_sz == 16 && sz == 32) {
         /* Apple's native ushort popcount writes a ushort.  Count(16 bits)
          * is at most 16, so widen its exact result after the operation. */
         agx_index count16 = agx_temp(b->shader, AGX_SIZE_16);
         agx_popcount_to(b, count16, s0);
         return agx_mov_to(b, dst, count16);
      }
      return agx_popcount_to(b, dst, s0);
   case nir_op_ufind_msb:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          src_sz == 32)
         return agx_emit_g17p_ufind_msb(b, dst, s0);
      return agx_ffs_to(b, dst, s0);
      BINOP(iand, and);
      BINOP(ior, or);
      BINOP(ixor, xor);
   case nir_op_interleave:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
          b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C)
         return agx_emit_g17p_interleave(b, dst, s0, s1);
      return agx_intl_to(b, dst, s0, s1);

   case nir_op_fadd:
      if (instr->def.bit_size == 16)
         return agx_hadd_to(b, dst, agx_coerce_f16_src(b, s0),
                            agx_coerce_f16_src(b, s1));
      else {
         agx_instr *add = agx_fadd_to(b, dst, agx_g17p_coerce_f32_src(b, s0),
                                     agx_g17p_coerce_f32_src(b, s1));
         add->no_signed_zero = !nir_alu_instr_is_signed_zero_preserve(instr);
         return add;
      }

   case nir_op_fmul:
      if (instr->def.bit_size == 16)
         return agx_hmul_to(b, dst, agx_coerce_f16_src(b, s0),
                            agx_coerce_f16_src(b, s1));
      else
         return agx_fmul_to(b, dst, agx_g17p_coerce_f32_src(b, s0),
                            agx_g17p_coerce_f32_src(b, s1));

   case nir_op_ffma:
      if (instr->def.bit_size == 16)
         return agx_hfma_to(b, dst, agx_coerce_f16_src(b, s0),
                            agx_coerce_f16_src(b, s1),
                            agx_coerce_f16_src(b, s2));
      else
         return agx_ffma_to(b, dst, agx_g17p_coerce_f32_src(b, s0),
                            agx_g17p_coerce_f32_src(b, s1),
                            agx_g17p_coerce_f32_src(b, s2));

   case nir_op_fsat: {
      agx_instr *I = agx_fadd_to(b, dst, s0, agx_negzero());
      if (instr->def.bit_size == 16)
         I->op = AGX_OPCODE_HADD;

      I->saturate = true;
      return I;
   }

   case nir_op_feq:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_EQ, false);
   case nir_op_flt:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_LT, false);
   case nir_op_fge:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_GE, false);
   case nir_op_fneu:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_EQ, true);

   case nir_op_ieq:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_UEQ, false);
   case nir_op_ine:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_UEQ, true);
   case nir_op_ilt:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_SLT, false);
   case nir_op_ige:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_SLT, true);
   case nir_op_ult:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_ULT, false);
   case nir_op_uge:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_ULT, true);

   case nir_op_inot:
      if (sz == 1)
         return agx_xor_to(b, dst, s0, i1);
      else
         return agx_not_to(b, dst, s0);

   case nir_op_b2b1:
      return agx_icmp_to(b, dst, s0, i0, AGX_ICOND_UEQ, true);

   case nir_op_fsqrt:
      if (instr->def.bit_size == 16) {
         /* G17P computes a scalar half sqrt as a half-input sqrt step into a
          * float32 temporary followed by a mixed multiply rounded once to
          * the half destination.  Keeping the temporary wide matches Apple
          * and avoids treating the neighbouring half as input state. */
         agx_index step = agx_temp(b->shader, AGX_SIZE_32);
         agx_srsqrt_to(b, step, s0);
         return agx_fmul_to(b, dst, step, s0);
      }
      return agx_fmul_to(b, dst, s0, agx_srsqrt(b, s0));
   case nir_op_fabs:
      return agx_fmov_to(b, dst, agx_abs(s0));
   case nir_op_fneg:
      return agx_fmov_to(b, dst, agx_neg(s0));

   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_fmin_agx:
   case nir_op_fmax_agx:
      return agx_fminmax_to(b, dst, s0, s1, instr);

   case nir_op_iabs: {
      assert((sz == 16 || sz == 32) && "8-bit absolute values lowered");
      /* Keep G17P's absolute value independent of signed isel.  The usual
       * two's-complement identity is exact even for the most negative value,
       * and a 16-bit value uses the same arithmetic shift, XOR and subtract
       * at half width. */
      agx_index sign = agx_asr(b, s0, agx_immediate(sz - 1));
      agx_index magnitude = agx_xor(b, s0, sign);
      return agx_iadd_to(b, dst, magnitude, agx_neg(sign), 0);
   }

   case nir_op_imin:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_SLT);
   case nir_op_imax:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_SGT);
   case nir_op_umin:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_ULT);
   case nir_op_umax:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_UGT);
   case nir_op_bounds_agx:
      /* end offset > bound ? 0 : data */
      return agx_icmpsel_to(b, dst, s1, s2, agx_zero(), s0, AGX_ICOND_UGT);

   case nir_op_iadd:
      return agx_iadd_to(b, dst, s0, s1, 0);
   case nir_op_imadshl_agx:
      return agx_emit_imadshl_agx(b, instr, dst, s0, s1, s2, s3);
   case nir_op_imsubshl_agx:
      return agx_emit_imadshl_agx(b, instr, dst, s0, s1, agx_neg(s2), s3);
   case nir_op_isub:
      /* The native G17P subtract form takes two registers. Do not route an
       * immediate minuend through iadd's narrow-source sign-extension bit:
       * that bit is not arithmetic negation and would turn `c - x` into
       * `c + x`.
       */
      if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
          sz == 32 && s0.type == AGX_INDEX_IMMEDIATE)
         s0 = agx_mov_imm(b, 32, s0.value);
      return agx_iadd_to(b, dst, s0, agx_neg(s1), 0);
   case nir_op_ineg:
      /* G17P's iadd operand modifier is a narrow-source sign extension, not
       * an arithmetic negate. Materialize the zero minuend so post-RA
       * instruction selection reaches the native register-register isub
       * form instead of silently encoding `0 + x`.
       */
      if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
          sz == 32) {
         agx_index zero = agx_mov_imm(b, 32, 0);
         return agx_iadd_to(b, dst, zero, agx_neg(s0), 0);
      }
      return agx_iadd_to(b, dst, i0, agx_neg(s0), 0);
   case nir_op_imul:
      return agx_imad_to(b, dst, s0, s1, i0, 0);
   case nir_op_umul_2x32_64:
      return agx_imad_to(b, dst, agx_abs(s0), agx_abs(s1), i0, 0);
   case nir_op_imul_2x32_64:
      return agx_imad_to(b, dst, s0, s1, i0, 0);
   case nir_op_umul_high:
      return agx_mul_high_to(b, dst, s0, s1, false);
   case nir_op_imul_high:
      return agx_mul_high_to(b, dst, s0, s1, true);

   case nir_op_ishl:
      /* NIR masks a half-value shift count to its low four bits. An
       * explicit low-half SSA value therefore preserves every defined count,
       * and selects native905's witnessed half-count G18G encoding.
       * Subdivide before RA; do not reinterpret a live full register later. */
      if (agx_is_g18g(&b->shader->key->dev) &&
          sz == 16 && s1.size == AGX_SIZE_32 &&
          s1.type != AGX_INDEX_IMMEDIATE) {
         agx_index half_count = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, half_count, s1, 0);
         s1 = half_count;
      }
      /* Native G17P implements constant left shifts 1..4 with the iadd
       * source-B shift and 5..8 with two such adds (shifting by k-4, then 4),
       * using SLL only from 9 (captured shift corpus sll_1..8, and the native
       * throughput shader's x << 5). The adds take the zero immediate in
       * slot A. G18G keeps the shift instruction: its iadd encoder has no
       * shifted-source form.
       */
      if (agx_isa_is_agx3(b->shader->key->dev.encoding) &&
          !agx_is_g18g(&b->shader->key->dev) &&
          sz == 32 && nir_src_is_const(instr->src[1].src)) {
         const unsigned shift = nir_alu_src_as_uint(instr->src[1]) & 31;
         if (shift >= 1 && shift <= 4)
            return agx_iadd_to(b, dst, agx_zero(), s0, shift);
         if (shift >= 5 && shift <= 8) {
            agx_index partial = agx_temp(b->shader, AGX_SIZE_32);
            agx_iadd_to(b, partial, agx_zero(), s0, shift - 4);
            return agx_iadd_to(b, dst, agx_zero(), partial, 4);
         }
      }
      return agx_bfi_to(b, dst, i0, s0, s1, 0);
   case nir_op_ushr: {
      /* NIR masks a half-value shift count to its low four bits. An
       * explicit low-half SSA value therefore preserves every defined count,
       * and selects native905's witnessed half-count G18G encoding.
       * Subdivide before RA; do not reinterpret a live full register later. */
      if (agx_is_g18g(&b->shader->key->dev) &&
          sz == 16 && s1.size == AGX_SIZE_32 &&
          s1.type != AGX_INDEX_IMMEDIATE) {
         agx_index half_count = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, half_count, s1, 0);
         s1 = half_count;
      }
      /* AGX3B only (this and the SRL count path below): the G17P shift
       * boundary materialization emits IADD forms AGX3C cannot encode. */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          sz == 32 && nir_src_is_const(instr->src[1].src)) {
         const unsigned shift = nir_alu_src_as_uint(instr->src[1]) & 31;
         if (shift == 0)
            return agx_mov_to(b, dst, s0);
         return agx_ushr_to(b, dst, s0, agx_immediate(shift));
      }
      /* G17P's register `srl` takes the ordinary positive count, as
       * applegpu_g17p's `ushort(a >> s)` (a710540002080000d0130500, after
       * its own `s & 31`) does. This used to build `0 - count`, but the
       * G17P iadd packer read NEG as a sign extension and emitted `0 +
       * count`; every hardware-proven program therefore already carries
       * the positive count through this add, which is kept. */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         if (s1.type == AGX_INDEX_IMMEDIATE) {
            /* The immediate SRL form consumes the positive literal itself. */
            if (!(sz == 32 && s1.value >= 1 && s1.value <= 31))
               s1 = agx_g17p_shift_boundary(b, s1.value);
         } else {
            agx_index boundary = agx_temp(b->shader, AGX_SIZE_32);
            agx_iadd_to(b, boundary, i0, s1, 0);
            s1 = boundary;
         }
      }
      return agx_ushr_to(b, dst, s0, s1);
   }
   case nir_op_ishr:
      return agx_asr_to(b, dst, s0, s1);

   case nir_op_extr_agx:
      return agx_extr_to(b, dst, s0, s1, s2,
                         nir_alu_src_as_uint(instr->src[3]));

   case nir_op_ubitfield_extract: {
      unsigned m = nir_alu_src_as_uint(instr->src[2]);
      assert(m != 0 && "should've been optimized");

      /* Disable masking if the whole thing is used */
      if (m >= 32)
         m = 0;

      return agx_bfeil_to(b, dst, i0, s0, s1, m);
   }

   case nir_op_bcsel:
      return agx_icmpsel_to(b, dst, s0, i0, s2, s1, AGX_ICOND_UEQ);

   case nir_op_i2i32: {
      if (src_sz == 8) {
         if (agx_isa_is_agx3(b->shader->key->dev.encoding)) {
            /* G17P's grounded shifts operate on complete 32-bit registers.
             * Widen first and sign-extend from bit 7 as Apple does, rather
             * than inventing a half-register BFI/ASR encoding. */
            agx_index wide = agx_pad_to_32(b, s0);
            agx_index zero32 = agx_mov_imm(b, 32, 0);
            agx_index shift = agx_mov_imm(b, 32, 24);
            agx_index shifted =
               agx_bfi(b, zero32, wide, shift, 0);
            return agx_asr_to(b, dst, shifted, shift);
         }

         /* Sign extend in software, NIR likes 8-bit conversions */
         agx_index ishl16 = agx_bfi(b, i0, s0, agx_immediate(8), 0);
         return agx_asr_to(b, dst, ishl16, agx_immediate(8));
      } else {
         assert(s0.size == AGX_SIZE_16 && "other conversions lowered");
         return agx_signext_to(b, dst, s0);
      }
   }

   case nir_op_i2i16: {
      if (src_sz == 8) {
         if (agx_isa_is_agx3(b->shader->key->dev.encoding)) {
            /* Perform the same exact 32-bit sign extension and expose its
             * low half only after the shift pair has completed. */
            agx_index wide = agx_pad_to_32(b, s0);
            agx_index zero32 = agx_mov_imm(b, 32, 0);
            agx_index shift = agx_mov_imm(b, 32, 24);
            agx_index shifted =
               agx_bfi(b, zero32, wide, shift, 0);
            agx_index extended = agx_asr(b, shifted, shift);
            return agx_subdivide_to(b, dst, extended, 0);
         }

         /* Sign extend in software, NIR likes 8-bit conversions */
         agx_index ishl16 = agx_bfi(b, i0, s0, agx_immediate(8), 0);
         return agx_asr_to(b, dst, ishl16, agx_immediate(8));
      } else {
         assert(s0.size == AGX_SIZE_32 && "other conversions lowered");
         return agx_subdivide_to(b, dst, s0, 0);
      }
   }

   case nir_op_u2u16: {
      if (s0.size == AGX_SIZE_32)
         return agx_subdivide_to(b, dst, s0, 0);
      else
         return agx_mov_to(b, dst, s0);
   }

   /* It will be put into a 16-bit register, but zero out the garbage. We could
    * optimize this in the future but it ensures correctness for u2u16(u2u8(x))
    * sequences.
    */
   case nir_op_u2u8:
   case nir_op_i2i8:
      return agx_and_to(b, dst, s0, agx_immediate(0xFF));

   case nir_op_iadd_sat: {
      agx_instr *I = agx_iadd_to(b, dst, s0, s1, 0);
      I->saturate = true;
      return I;
   }

   case nir_op_isub_sat: {
      agx_instr *I = agx_iadd_to(b, dst, s0, agx_neg(s1), 0);
      I->saturate = true;
      return I;
   }

   case nir_op_uadd_sat: {
      agx_instr *I = agx_iadd_to(b, dst, agx_abs(s0), agx_abs(s1), 0);
      I->saturate = true;
      return I;
   }

   case nir_op_usub_sat: {
      agx_instr *I = agx_iadd_to(b, dst, agx_abs(s0), agx_neg(agx_abs(s1)), 0);
      I->saturate = true;
      return I;
   }

   case nir_op_fsin_agx: {
      if (agx_isa_is_agx3(b->shader->key->dev.encoding)) {
         /* Compatibility for callers which already performed the AGX2 NIR
          * quadrant reduction. Fold [0,4] exactly to [-1,1], then use the
          * same native sinc operation as the G17P radians lowering. The
          * folds stay in the operand's own precision: the builder would size
          * the temporary from the leading immediate and make a 16-bit add of
          * a 32-bit quadrant, which neither keeps the precision nor packs. */
         /* The native sinc is a 32-bit funop with no half form, so a
          * mediump quadrant is evaluated at full precision and narrowed. */
         const bool half = s0.size == AGX_SIZE_16;
         agx_index x = s0;
         if (half) {
            x = agx_temp(b->shader, AGX_SIZE_32);
            agx_fmov_to(b, x, s0);
         }
         agx_index middle = agx_temp(b->shader, x.size);
         agx_fadd_to(b, middle, agx_immediate_f(2.0f), agx_neg(x));
         agx_index last = agx_fadd(b, x, agx_immediate_f(-4.0f));
         agx_index t = agx_fcmpsel(b, x, agx_immediate_f(1.0f), x,
                                   middle, AGX_FCOND_LE);
         t = agx_fcmpsel(b, x, agx_immediate_f(3.0f), last, t,
                         AGX_FCOND_GT);
         if (!half)
            return agx_fmul_to(b, dst, agx_sinc_agx3(b, t), t);
         return agx_fmov_to(b, dst, agx_fmul(b, agx_sinc_agx3(b, t), t));
      }

      agx_index fixup = agx_sin_pt_1(b, s0);
      agx_index sinc = agx_sin_pt_2(b, fixup);
      return agx_fmul_to(b, dst, sinc, fixup);
   }

   case nir_op_bittest_select_agx3: {
      assert(nir_src_is_const(instr->src[1].src));
      const unsigned mask = nir_alu_src_as_uint(instr->src[1]);
      assert(mask == 1 || mask == 2 || mask == 4);
      return agx_bittest_select_agx3_to(b, dst, s0, s2, s3, mask);
   }

   case nir_op_f2i16:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_S16), s0,
                            AGX_ROUND_RTZ);

   case nir_op_f2i32:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_S32), s0,
                            AGX_ROUND_RTZ);

   case nir_op_f2u16:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_U16), s0,
                            AGX_ROUND_RTZ);

   case nir_op_f2u32:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_U32), s0,
                            AGX_ROUND_RTZ);

   case nir_op_u2f16:
   case nir_op_u2f32: {
      if (src_sz == 64)
         UNREACHABLE("64-bit conversions unimplemented");

      /* As for i2f below: the unsigned-byte float32 conversion reads only
       * the low eight bits of its half (agx_nir_fold_ubyte_to_float). */
      if (src_sz == 8 && sz == 32 &&
          b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          !agx_is_g18g(&b->shader->key->dev)) {
         nir_alu_instr *narrow = nir_src_as_alu(instr->src[0].src);
         if (narrow &&
             (narrow->op == nir_op_u2u8 || narrow->op == nir_op_i2i8) &&
             nir_src_bit_size(narrow->src[0].src) == 16)
            return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_U8_TO_F),
                                  agx_alu_src_index(b, narrow->src[0]),
                                  AGX_ROUND_RTE);
      }

      enum agx_convert mode = (src_sz == 32)   ? AGX_CONVERT_U32_TO_F
                              : (src_sz == 16) ? AGX_CONVERT_U16_TO_F
                                               : AGX_CONVERT_U8_TO_F;

      return agx_convert_to(b, dst, agx_immediate(mode), s0, AGX_ROUND_RTE);
   }

   case nir_op_i2f16:
   case nir_op_i2f32: {
      if (src_sz == 64)
         UNREACHABLE("64-bit conversions unimplemented");

      /* The signed-byte conversion reads only the low eight bits of its half
       * register, so the byte of a u2u8/i2i8 of a half needs no mask
       * (agx_nir_fold_byte_to_float). */
      if (src_sz == 8 &&
          b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          !agx_is_g18g(&b->shader->key->dev)) {
         nir_alu_instr *narrow = nir_src_as_alu(instr->src[0].src);
         if (narrow &&
             (narrow->op == nir_op_u2u8 || narrow->op == nir_op_i2i8) &&
             nir_src_bit_size(narrow->src[0].src) == 16)
            return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_S8_TO_F),
                                  agx_alu_src_index(b, narrow->src[0]),
                                  AGX_ROUND_RTE);
      }

      enum agx_convert mode = (src_sz == 32)   ? AGX_CONVERT_S32_TO_F
                              : (src_sz == 16) ? AGX_CONVERT_S16_TO_F
                                               : AGX_CONVERT_S8_TO_F;

      return agx_convert_to(b, dst, agx_immediate(mode), s0, AGX_ROUND_RTE);
   }

   case nir_op_pack_32_2x16_split:
   case nir_op_pack_64_2x32_split: {
      agx_index idx[] = {s0, s1};
      return agx_emit_collect_to(b, dst, 2, idx);
   }

   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_32_2x16_split_x:
      return agx_subdivide_to(b, dst, s0, 0);

   case nir_op_unpack_64_2x32_split_y:
   case nir_op_unpack_32_2x16_split_y:
      return agx_subdivide_to(b, dst, s0, 1);

   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      agx_index idx[] = {s0, s1, s2, s3};
      return agx_emit_collect_to(b, dst, srcs, idx);
   }

   case nir_op_vec8:
   case nir_op_vec16: {
      agx_index idx[NIR_MAX_VEC_COMPONENTS];
      assert(srcs <= ARRAY_SIZE(idx));
      for (unsigned i = 0; i < srcs; ++i)
         idx[i] = agx_alu_src_index(b, instr->src[i]);
      return agx_emit_collect_to(b, dst, srcs, idx);
   }

   default:
      /* For instance 64-bit float conversions, which need shaderFloat64.
       * Refuse the shader as for an unhandled intrinsic. */
      fprintf(stderr, "Unhandled ALU op %s\n", nir_op_infos[instr->op].name);
      b->shader->unsupported_instruction = true;
      return NULL;
   }
}

static enum agx_lod_mode
agx_lod_mode_for_nir(nir_texop op, bool biased, bool min_lod, bool lod_is_zero)
{
   switch (op) {
   case nir_texop_tex:
   case nir_texop_tg4:
      /* We could support this for tex, but it's never actually seen because tex
       * is always turned into txb to implement sampler LOD bias in Vulkan.
       */
      assert(!min_lod && "unimplemented");

      return AGX_LOD_MODE_AUTO_LOD;
   case nir_texop_txb:
      return min_lod ? AGX_LOD_MODE_AUTO_LOD_BIAS_MIN
                     : AGX_LOD_MODE_AUTO_LOD_BIAS;
   case nir_texop_lod:
      assert(!min_lod);
      return biased ? AGX_LOD_MODE_AUTO_LOD_BIAS : AGX_LOD_MODE_AUTO_LOD;
   case nir_texop_txd:
      return min_lod ? AGX_LOD_MODE_LOD_GRAD_MIN : AGX_LOD_MODE_LOD_GRAD;
   case nir_texop_txl:
   case nir_texop_txf:
      assert(!min_lod);
      return lod_is_zero ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_LOD_MIN;
   case nir_texop_txf_ms:
      assert(!min_lod);
      assert(lod_is_zero && "no mipmapping");
      return AGX_LOD_MODE_AUTO_LOD;
   default:
      UNREACHABLE("Unhandled texture op");
   }
}

static enum agx_gather
agx_gather_for_nir(nir_tex_instr *tex)
{
   if (tex->op == nir_texop_tg4) {
      enum agx_gather components[] = {
         AGX_GATHER_R,
         AGX_GATHER_G,
         AGX_GATHER_B,
         AGX_GATHER_A,
      };

      assert(tex->component < ARRAY_SIZE(components));
      return components[tex->component];
   } else {
      return AGX_GATHER_NONE;
   }
}

static void
agx_emit_tex(agx_builder *b, nir_tex_instr *instr)
{
   agx_index coords = agx_null(), bindless = agx_immediate(0),
             texture = agx_immediate(instr->texture_index),
             sampler = agx_immediate(instr->sampler_index),
             lod = agx_immediate(0), compare = agx_null(),
             packed_offset = agx_null();

   /* Default to the txf sampler at ss0 */
   if (!nir_tex_instr_need_sampler(instr)) {
      sampler = agx_immediate(0);
   }

   bool lod_is_zero = true;
   bool reused_lod = false;

   for (unsigned i = 0; i < instr->num_srcs; ++i) {
      agx_index index = agx_src_index(&instr->src[i].src);

      switch (instr->src[i].src_type) {
      case nir_tex_src_backend1:
         coords = index;
         break;

      case nir_tex_src_backend2:
         /* G17P's texture offset is an immediate in SMP, with signed
          * x/y nibbles and a third z nibble for 3D. Keep the constant from
          * agx_nir_lower_texture in that form instead of materializing a GPR
          * that the native instruction does not read.
          */
         /* AGX3C (G17G witnesses, texwit3): the same immediate f43/f51/f141
          * offset on 2D, 2D-array, 3D, comparison and gather samples, in every
          * LOD mode. Cubes have no offset form. G18G keeps j873's admission
          * of every sampler dimension. */
         const bool m5_offset =
            b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
            (instr->sampler_dim == GLSL_SAMPLER_DIM_2D ||
             instr->sampler_dim == GLSL_SAMPLER_DIM_3D);
         if ((b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
              m5_offset || agx_is_g18g(&b->shader->key->dev)) &&
             nir_src_is_const(instr->src[i].src) &&
             (instr->op == nir_texop_tg4 ||
              nir_src_as_uint(instr->src[i].src) <= 0xfff)) {
            uint32_t value = nir_src_as_uint(instr->src[i].src);
            /* Only G17P's lowering keeps gather X/Y in separate 16-bit lanes;
             * C (generation 10) already packed the compact (y << 4) | x. */
            if (instr->op == nir_texop_tg4 &&
                b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
               value = (value & 0xf) | ((value >> 12) & 0xf0);
            /* Apple's compiler emits an all-zero offset as a plain sample. */
            packed_offset = value ? agx_immediate(value) : agx_null();
         } else {
            packed_offset = index;
         }
         if (packed_offset.type == AGX_INDEX_IMMEDIATE)
            packed_offset.size = AGX_SIZE_32;
         break;

      case nir_tex_src_lod:
      case nir_tex_src_bias:
         reused_lod = !list_is_singular(&instr->src[i].src.ssa->uses);
         /* G17P has an exact immediate texel-fetch level encoding. Keep
          * compile-time levels in that form instead of promoting them into
          * the per-lane PACK_SMPIR path. Besides being smaller, this matches
          * Apple's constant read(..., level) programs and avoids introducing
          * an asynchronous staging dependency for a literal. */
         if ((b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
              b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
             instr->op == nir_texop_txf &&
             nir_src_is_const(instr->src[i].src) &&
             nir_src_as_uint(instr->src[i].src) <= 31)
            lod = agx_immediate(nir_src_as_uint(instr->src[i].src));
         else
            lod = index;
         lod_is_zero = nir_src_is_const(instr->src[i].src) &&
                       nir_src_as_uint(instr->src[i].src) == 0;
         break;

      case nir_tex_src_lod_bias_min_agx:
         assert(index.size == AGX_SIZE_32);
         lod = index;
         break;

      case nir_tex_src_comparator:
         assert(index.size == AGX_SIZE_32);
         compare = index;
         break;

      case nir_tex_src_texture_offset:
         texture = index;
         break;
      case nir_tex_src_sampler_handle:
         /* AGX2 preamble lowering replaces unpromoted bindless sampler
          * tokens with their heap indices. Custom-border clamp-to-0 samples
          * also carry plain indices, so the intrinsic is not evidence that
          * the heap is unused. Keep the upstream AGX2 binding contract;
          * native AGX3 samplers retain their separate resource path.
          */
         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
            b->shader->out->uses_sampler_heap = true;

         nir_intrinsic_instr *intr = nir_src_as_intrinsic(instr->src[i].src);
         if (intr && intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
            b->shader->out->uses_sampler_heap = true;
            /* Heap offset */
            sampler = agx_src_index(&intr->src[1]);
         } else {
            sampler = index;
         }
         break;

      case nir_tex_src_texture_handle:
         if (instr->backend_flags & AGX_TEXTURE_FLAG_TABLE_INDEX) {
            /* The handle fed the descriptor crawl; the operation itself is
             * bindful, as G17G's native fragment fetches are. */
            texture = agx_immediate(instr->texture_index);
            break;
         }
         texture =
            agx_translate_bindless_handle(b, &instr->src[i].src, &bindless);
         break;

      case nir_tex_src_min_lod:
         assert(instr->op == nir_texop_txd && "other cases lowered");
         break;

      case nir_tex_src_ddx: {
         int y_idx = nir_tex_instr_src_index(instr, nir_tex_src_ddy);
         assert(y_idx >= 0 && "we only handle gradients");

         int min_idx = nir_tex_instr_src_index(instr, nir_tex_src_min_lod);
         bool has_min = min_idx >= 0;
         agx_index min;

         unsigned n = nir_tex_instr_src_size(instr, y_idx);
         assert((n == 2 || n == 3) && "other sizes not supported");

         agx_index index2 = agx_src_index(&instr->src[y_idx].src);

         if (has_min) {
            min = agx_src_index(&instr->src[min_idx].src);

            /* Undef extend to 32-bit since our IR is iffy */
            min = agx_pad_to_32(b, min);
         }

         /* We explicitly don't cache about the split cache for this */
         unsigned chans = (2 * n) + (has_min ? 1 : 0);
         lod = agx_vec_temp(b->shader, AGX_SIZE_32, chans);
         agx_instr *I = agx_collect_to(b, lod, chans);

         for (unsigned i = 0; i < n; ++i) {
            I->src[(2 * i) + 0] = agx_emit_extract(b, index, i);
            I->src[(2 * i) + 1] = agx_emit_extract(b, index2, i);
         }

         if (has_min)
            I->src[2 * n] = min;

         break;
      }

      case nir_tex_src_ddy:
         /* handled above */
         break;

      default:
         UNREACHABLE("Unexpected texture source");
      }
   }

   bool has_min_lod =
      nir_tex_instr_src_index(instr, nir_tex_src_lod_bias_min_agx) >= 0 ||
      nir_tex_instr_src_index(instr, nir_tex_src_min_lod) >= 0;

   bool has_bias = (has_min_lod && instr->op != nir_texop_txd) ||
                   (nir_tex_instr_src_index(instr, nir_tex_src_bias) >= 0);

   enum agx_lod_mode lod_mode =
      agx_lod_mode_for_nir(instr->op, has_bias, has_min_lod, lod_is_zero);

   /* AGX2's AUTO_LOD encoding also covers txl(..., 0). AGX3's AUTO form is
    * specifically the fragment derivative form, while Apple's compute and
    * vertex witnesses use the explicit immediate-level encoding. Keep every
    * non-fragment txl at level zero explicit so the packer selects that
    * native form. */
   /* G18G also keeps a fragment txl/txf at level zero explicit (j873). */
   if (((b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
         b->shader->stage != MESA_SHADER_FRAGMENT) ||
        agx_is_g18g(&b->shader->key->dev)) &&
       (instr->op == nir_texop_txl || instr->op == nir_texop_txf) &&
       lod_is_zero) {
      lod_mode = AGX_LOD_MODE_LOD_MIN;
      /* The native G17P vertex form carries level zero in the instruction.
       * Keeping NIR's fp32 zero SSA here made texture_sample source 1 two
       * half-registers wide even though the explicit-level IR operand is one,
       * so validation failed before the packer could select that form. */
      lod = agx_immediate(0);
   }

   /* Rectangle textures have one level. Their sampler-bias lowering may
    * still produce txb outside a fragment shader, where no derivative quad
    * exists. Native G18 pixel-coordinate vertex/compute samples use an
    * explicit zero level; keep the unnormalized coordinates unchanged. */
   if (b->shader->key->dev.emit_agx3d &&
       b->shader->stage != MESA_SHADER_FRAGMENT &&
       instr->sampler_dim == GLSL_SAMPLER_DIM_RECT &&
       (instr->op == nir_texop_tex || instr->op == nir_texop_txb)) {
      lod_mode = AGX_LOD_MODE_LOD_MIN;
      lod = agx_immediate(0);
   }

   if (lod_mode == AGX_LOD_MODE_AUTO_LOD) {
      /* Ignored logically but asserted 0 */
      lod = agx_immediate(0);
   }

   /* Reused G17G parameters are packed once in the normal register file.
    * CSE shares identical pure producers and liveness preserves the word
    * until its last publication. Single-use half-floats retain the direct
    * staging path. */
   const bool packed_lod =
      b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
      !agx_is_g18g(&b->shader->key->dev) &&
      reused_lod && lod.type != AGX_INDEX_IMMEDIATE &&
      lod.size == AGX_SIZE_16 &&
      instr->op != nir_texop_txf && instr->op != nir_texop_txf_ms &&
      (instr->sampler_dim == GLSL_SAMPLER_DIM_2D ||
       instr->sampler_dim == GLSL_SAMPLER_DIM_3D) && !instr->is_array &&
      !has_min_lod && (lod_mode == AGX_LOD_MODE_LOD_MIN ||
                      lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS);
   if (packed_lod) {
      agx_index word = agx_temp(b->shader, AGX_SIZE_32);
      agx_pack_sampler_lod_agx3_to(b, word, lod);
      lod = word;
   }

   agx_index dst = agx_def_index(&instr->def);

   /* Pack the shadow reference and offset sources into the one texture IR
    * operand. G17P constants leave only the comparator here because their
    * offset rides in SMP. Runtime G17P offsets remain a two-component vector
    * for PACK_SMPIR; a comparison, when present, is its leading component. */
   agx_index compare_offset = agx_null();

   /* G17P constant offsets live in the instruction, leaving src[5] to
    * the comparator. Runtime gather offsets retain the combined vector;
    * the packer publishes each value into its own native operand bank. */
   const bool offset_present = !agx_is_null(packed_offset);
   const uint16_t offset_xyz =
      packed_offset.type == AGX_INDEX_IMMEDIATE
         ? (uint16_t)packed_offset.value : 0;

   const bool g17p_split_offset =
      agx_isa_is_agx3(b->shader->key->dev.encoding) &&
      packed_offset.type == AGX_INDEX_IMMEDIATE;

   if (g17p_split_offset && !agx_is_null(compare))
      packed_offset = agx_null();

   if (!agx_is_null(compare) && !agx_is_null(packed_offset)) {
      /* A 32-bit immediate inside a COLLECT is not a shape
       * agx_validate_sources() models, and it exits the process rather than
       * failing the shader closed. */
      if (packed_offset.type == AGX_INDEX_IMMEDIATE)
         packed_offset = agx_mov_imm(b, 32, packed_offset.value);

      const unsigned offset_channels = agx_channels(packed_offset);
      agx_index sources[4] = {compare};
      assert(offset_channels <= 3);
      if (offset_channels == 1) {
         sources[1] = packed_offset;
      } else {
         agx_index offset_components[3] = {
            agx_null(), agx_null(), agx_null()
         };
         /* Vector loads are contiguous but do not necessarily have an entry
          * in allocated_vec. Split this texture operand explicitly instead
          * of assuming its producer was an agx_emit_collect_to(). The split
          * remains a pseudo instruction and coalesces with a preceding
          * collect when one exists. */
         agx_emit_split(b, offset_components, packed_offset, offset_channels);
         for (unsigned c = 0; c < offset_channels; ++c)
            sources[c + 1] = offset_components[c];
      }
      compare_offset = agx_emit_collect(b, offset_channels + 1, sources);
   }
   else if (!agx_is_null(packed_offset))
      compare_offset = packed_offset;
   else if (!agx_is_null(compare))
      compare_offset = compare;

   /* AGX3 encodes texture and sampler table indices up to 15 as immediates;
    * a larger constant index (a descriptor heap offset) takes the register
    * form, a 64-bit index. */
   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
      if (texture.type == AGX_INDEX_IMMEDIATE && texture.value > 15)
         texture = agx_mov_imm(b, 64, texture.value);
      if (sampler.type == AGX_INDEX_IMMEDIATE && sampler.value > 15)
         sampler = agx_mov_imm(b, 64, sampler.value);
   }

   agx_index tmp = agx_vec_temp(b->shader, dst.size, instr->is_sparse ? 8 : 4);
   agx_instr *I = agx_texture_sample_to(
      b, tmp, coords, lod, bindless, texture, sampler, compare_offset,
      agx_tex_dim(instr->sampler_dim, instr->is_array), lod_mode, 0,
      offset_present, !agx_is_null(compare),
      instr->op == nir_texop_lod, agx_gather_for_nir(instr), instr->is_sparse);

   /* The packer reads the G17P offset from the instruction, not from src[5],
    * so it survives being split away from the comparator above. */
   I->offset_xyz = offset_xyz;
   /* A C sample's packed parameter is a word, not an unpacked half-float. */
   I->memory_is_packed = packed_lod;

   if (instr->op == nir_texop_txf || instr->op == nir_texop_txf_ms) {
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      b->shader->out->uses_txf = true;
      b->shader->out->sampler_state_count =
         MAX2(b->shader->out->sampler_state_count, 1);
   }

   /* Destination masking doesn't seem to work properly for gathers (because
    * it's mostly pointless), but it does show up in the lowering of
    * textureGatherOffsets. Don't try to mask the destination for gathers.
    *
    * TODO: Check if it works with sparse.
    */
   bool masked = (instr->op != nir_texop_tg4) && !instr->is_sparse;
   if (I->op == AGX_OPCODE_TEXTURE_LOAD)
      masked = agx_fetch_masked(b, &instr->def, masked);
   I->mask = agx_expand_tex_to(b, &instr->def, tmp, masked);
}

static void
agx_emit_jump(agx_builder *b, nir_jump_instr *instr)
{
   if (instr->type == nir_jump_halt) {
      agx_stop(b);
   } else {
      assert(instr->type == nir_jump_break);
      agx_block_add_successor(b->shader->current_block, b->shader->break_block);
      agx_break(b, b->shader->loop_nesting + 1, b->shader->break_block);
   }

   b->shader->current_block->unconditional_jumps = true;
}

static void
agx_emit_phi(agx_builder *b, nir_phi_instr *instr)
{
   agx_instr *I =
      agx_phi_to(b, agx_def_index(&instr->def), exec_list_length(&instr->srcs));

   /* Deferred */
   I->phi = instr;
}

/* Look up the AGX block corresponding to a given NIR block. Used when
 * translating phi nodes after emitting all blocks.
 */
static agx_block *
agx_from_nir_block(agx_context *ctx, nir_block *block)
{
   return ctx->indexed_nir_blocks[block->index];
}

static void
agx_emit_phi_deferred(agx_context *ctx, agx_block *block, agx_instr *I)
{
   nir_phi_instr *phi = I->phi;
   I->phi = NULL;

   /* Guaranteed by lower_phis_to_scalar */
   assert(phi->def.num_components == 1);

   nir_foreach_phi_src(src, phi) {
      agx_block *pred = agx_from_nir_block(ctx, src->pred);
      unsigned i = agx_predecessor_index(block, pred);
      assert(i < I->nr_srcs);

      I->src[i] = agx_src_index(&src->src);
   }
}

static void
agx_emit_phis_deferred(agx_context *ctx)
{
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, I)
         agx_emit_phi_deferred(ctx, block, I);
   }
}

static void
agx_emit_undef(agx_builder *b, nir_undef_instr *instr)
{
   /* For now, just lower undefs to zero. This doesn't matter too much, since
    * the lowering happens in NIR and this just allows for late lowering passes
    * to result in undefs.
    */
   if (instr->def.num_components > 1) {
      agx_index zeros[NIR_MAX_VEC_COMPONENTS];
      assert(instr->def.num_components <= ARRAY_SIZE(zeros));
      agx_index zero = agx_mov_imm(b, instr->def.bit_size, 0);
      for (unsigned i = 0; i < instr->def.num_components; ++i)
         zeros[i] = zero;
      agx_emit_collect_to(b, agx_def_index(&instr->def),
                          instr->def.num_components, zeros);
   } else {
      agx_mov_imm_to(b, agx_def_index(&instr->def), 0);
   }
}

static void
agx_emit_instr(agx_builder *b, struct nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
      agx_emit_load_const(b, nir_instr_as_load_const(instr));
      break;

   case nir_instr_type_intrinsic:
      agx_emit_intrinsic(b, nir_instr_as_intrinsic(instr));
      break;

   case nir_instr_type_alu:
      agx_emit_alu(b, nir_instr_as_alu(instr));
      break;

   case nir_instr_type_tex:
      agx_emit_tex(b, nir_instr_as_tex(instr));
      break;

   case nir_instr_type_jump:
      agx_emit_jump(b, nir_instr_as_jump(instr));
      break;

   case nir_instr_type_phi:
      agx_emit_phi(b, nir_instr_as_phi(instr));
      break;

   case nir_instr_type_undef:
      agx_emit_undef(b, nir_instr_as_undef(instr));
      break;

   default:
      UNREACHABLE("should've been lowered");
   }
}

static agx_block *
agx_create_block(agx_context *ctx)
{
   agx_block *blk = rzalloc(ctx, agx_block);

   /* This is conservative, TODO: divergence analysis */
   blk->divergent = ctx->total_nesting > 0;
   blk->cf_divergent = ctx->divergent_nesting > 0;

   util_dynarray_init(&blk->predecessors, blk);

   return blk;
}

static agx_block *
emit_block(agx_context *ctx, nir_block *block)
{
   if (ctx->after_block) {
      ctx->current_block = ctx->after_block;
      ctx->after_block = NULL;
   } else {
      ctx->current_block = agx_create_block(ctx);
   }

   agx_block *blk = ctx->current_block;
   list_addtail(&blk->link, &ctx->blocks);
   list_inithead(&blk->instructions);

   ctx->indexed_nir_blocks[block->index] = blk;

   agx_builder _b = agx_init_builder(ctx, agx_after_block(blk));

   nir_foreach_instr(instr, block) {
      /* After a refused instruction its value is undefined; stop selecting
       * instructions (the compile is refused once emission returns). */
      if (ctx->unsupported_instruction)
         break;
      agx_emit_instr(&_b, instr);
   }

   return blk;
}

static agx_block *emit_cf_list(agx_context *ctx, struct exec_list *list);

static void
emit_if(agx_context *ctx, nir_if *nif)
{
   agx_block *first_block = ctx->current_block;
   agx_builder _b = agx_init_builder(ctx, agx_after_block(first_block));
   agx_index cond = agx_src_index(&nif->condition);

   agx_instr *if_ = agx_if_icmp(&_b, cond, agx_zero(), 1, AGX_ICOND_UEQ, true,
                                NULL /* filled in later */);
   const bool divergent = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2 ||
                          nir_src_is_divergent(&nif->condition);
   ctx->loop_nesting++;
   ctx->total_nesting++;
   ctx->divergent_nesting += divergent;

   /* Emit the two subblocks. */
   agx_block *if_block = emit_cf_list(ctx, &nif->then_list);
   agx_block *end_then = ctx->current_block;

   _b.cursor = agx_after_block(ctx->current_block);

   agx_block *else_block = emit_cf_list(ctx, &nif->else_list);
   agx_block *end_else = ctx->current_block;

   /* If the "if" fails, we fallthrough to the else */
   if_->target = else_block;

   /* Insert an else instruction at the beginning of the else block. We use
    * "else_fcmp 0.0, 0.0, eq" as unconditional else, matching the blob.
    *
    * If it fails, we fall through to the logical end of the last else block.
    */
   _b.cursor = agx_before_block(else_block);
   agx_else_fcmp(&_b, agx_zero(), agx_zero(), 1, AGX_FCOND_EQ, false, end_else);

   ctx->after_block = agx_create_block(ctx);

   agx_block_add_successor(first_block, if_block);
   agx_block_add_successor(first_block, else_block);
   agx_block_add_successor(end_then, ctx->after_block);
   agx_block_add_successor(end_else, ctx->after_block);

   _b.cursor = agx_after_block(ctx->current_block);
   agx_pop_exec(&_b, 1);
   ctx->loop_nesting--;
   ctx->total_nesting--;
   ctx->divergent_nesting -= divergent;
}

static void
emit_loop(agx_context *ctx, nir_loop *nloop)
{
   assert(!nir_loop_has_continue_construct(nloop));
   /* We only track nesting within the innermost loop, so push and reset */
   unsigned pushed_nesting = ctx->loop_nesting;
   const bool divergent = ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2 ||
                          nir_loop_is_divergent(nloop);
   ctx->loop_nesting = 0;
   ctx->total_nesting++;
   ctx->divergent_nesting += divergent;
   ctx->out->has_loops = true;

   agx_block *popped_break = ctx->break_block;

   ctx->break_block = agx_create_block(ctx);
   agx_block *loop_header = agx_create_block(ctx);

   /* AGX3 loops always open two architectural exec-mask levels before their
    * body. Apple's top-level prodloop witness starts with the same
    * unconditional CNDST(2) used by nested loops, and the shared footer pops
    * those two levels. AGX2 needs the push only inside other control flow,
    * where threads may already be masked off (TODO: divergence analysis).
    */
   agx_builder _b = agx_init_builder(ctx, agx_after_block(ctx->current_block));
   const bool agx3 = ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
   const bool top_level_agx3 = agx3 && ctx->total_nesting == 1;
   if (top_level_agx3)
      agx_push_exec(&_b, 0);
   if (agx3 || ctx->total_nesting > 1)
      agx_push_exec(&_b, 1);

   /* Fallthrough to loop header */
   agx_block_add_successor(ctx->current_block, loop_header);

   /* Emit the loop */
   ctx->after_block = loop_header;
   loop_header->loop_header = true;
   agx_block *start_block = emit_cf_list(ctx, &nloop->body);

   _b.cursor = agx_after_block(ctx->current_block);
   agx_jmp_exec_any(&_b, start_block);
   agx_pop_exec(&_b, 1);
   if (top_level_agx3)
      agx_pop_exec(&_b, 1);
   agx_block_add_successor(ctx->current_block, loop_header);

   /* Pop off */
   ctx->after_block = ctx->break_block;
   ctx->break_block = popped_break;

   /* Update shader-db stats */
   ++ctx->loop_count;

   /* All nested control flow must have finished */
   assert(ctx->loop_nesting == 0);

   /* Restore loop nesting (we might be inside an if inside an outer loop) */
   ctx->loop_nesting = pushed_nesting;
   ctx->total_nesting--;
   ctx->divergent_nesting -= divergent;
}

/* Before the first control flow structure, the nesting counter needs to be
 * zeroed for correct operation. This only happens at most once, since by
 * definition this occurs at the end of the first block, which dominates the
 * rest of the program. */

static void
emit_first_cf(agx_context *ctx)
{
   if (ctx->any_cf)
      return;

   agx_builder _b = agx_init_builder(ctx, agx_after_block(ctx->current_block));
   agx_begin_cf(&_b);
   ctx->any_cf = true;
}

static agx_block *
emit_cf_list(agx_context *ctx, struct exec_list *list)
{
   agx_block *start_block = NULL;

   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block: {
         agx_block *block = emit_block(ctx, nir_cf_node_as_block(node));

         if (!start_block)
            start_block = block;

         break;
      }

      case nir_cf_node_if:
         emit_first_cf(ctx);
         emit_if(ctx, nir_cf_node_as_if(node));
         break;

      case nir_cf_node_loop:
         emit_first_cf(ctx);
         emit_loop(ctx, nir_cf_node_as_loop(node));
         break;

      default:
         UNREACHABLE("Unknown control flow");
      }
   }

   return start_block;
}

static void
agx_set_st_vary_final(agx_context *ctx)
{
   agx_foreach_instr_global_rev(ctx, I) {
      if (agx_is_vertex_store(I)) {
         I->last = true;
         return;
      }
   }

   /* If we got here, there was no varying written. We need to mark that. */
   agx_block *last_block = list_last_entry(&ctx->blocks, agx_block, link);
   agx_builder _b = agx_init_builder(ctx, agx_after_block_logical(last_block));
   agx_no_varyings(&_b);
}

static void
agx_calc_stats(agx_context *ctx, unsigned size, struct agx2_stats *stats)
{
   struct agx_cycle_estimate cycles = agx_estimate_cycles(ctx);
   uint32_t old_preamble_inst = stats->preamble_inst;

   *stats = (struct agx2_stats){
      .alu = cycles.alu,
      .fscib = cycles.f_scib,
      .ic = cycles.ic,
      .code_size = size,
      .gprs = ctx->max_reg,
      .uniforms = ctx->out->push_count,
      .scratch = ctx->scratch_size_B,
      .threads = agx_occupancy_for_register_count(ctx->max_reg).max_threads,
      .loops = ctx->loop_count,
      .preamble_inst = old_preamble_inst,
   };

   /* Count instructions */
   agx_foreach_instr_global(ctx, I) {
      stats->instrs++;

      if (I->op == AGX_OPCODE_STACK_STORE)
         stats->spills++;
      else if (I->op == AGX_OPCODE_STACK_LOAD)
         stats->fills++;
   }
}

static bool
agx_lower_sincos_filter(const nir_instr *instr, UNUSED const void *_)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_fsin || alu->op == nir_op_fcos;
}

/* Sine and cosine are implemented via the sin_pt_1 and sin_pt_2 opcodes for
 * heavy lifting. sin_pt_2 implements sinc in the first quadrant, expressed in
 * turns (sin (tau x) / x), while sin_pt_1 implements a piecewise sign/offset
 * fixup to transform a quadrant angle [0, 4] to [-1, 1]. The NIR opcode
 * fsin_agx models the fixup, sinc, and multiply to obtain sine, so we just
 * need to change units from radians to quadrants modulo turns. Cosine is
 * implemented by shifting by one quadrant: cos(x) = sin(x + tau/4).
 */

static nir_def *
agx_lower_sincos_impl(struct nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_mov_alu(b, alu->src[0], 1);

   if ((uintptr_t)data && x->bit_size <= 32) {
      /* Native G17P sin/cos use three non-overlapping pieces of 2/pi.
       * The magic rounding add retains the quadrant in its low two bits.
       * Keep the explicit FMAs and subtraction order, including low pieces.
       * Native trig-fast text hashes are retained with the focused tests. */
      const unsigned bits = x->bit_size;
      const uint32_t saved_ctrl = b->fp_math_ctrl;
      b->fp_math_ctrl = nir_fp_no_fast_math;
      x = nir_f2f32(b, x);
      nir_def *hi = nir_imm_float(b, uif(0x3f22f983));
      nir_def *magic = nir_imm_float(b, uif(0x4b400000));
      nir_def *quadrant = nir_ffma(b, x, hi, magic);
      nir_def *r = nir_ffma(b, x, hi, nir_fsub(b, magic, quadrant));
      r = nir_ffma(b, x, nir_imm_float(b, uif(0x32dc9c88)), r);
      r = nir_ffma(b, x, nir_imm_float(b, uif(0x25a94fe1)), r);
      nir_def *complement = nir_fsub(b, nir_imm_float(b, 1), nir_fabs(b, r));
      const bool cosine = alu->op == nir_op_fcos;
      if ((uintptr_t)data == 2) {
         nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, quadrant, 1), 0);
         nir_def *negative = nir_ine_imm(b, nir_iand_imm(b, quadrant, 2), 0);
         r = nir_bcsel(b, odd, cosine ? nir_fneg(b, r) : complement,
                       cosine ? complement : r);
         r = nir_bcsel(b, negative, nir_fneg(b, r), r);
      } else {
         r = nir_bittest_select_agx3(b, quadrant, nir_imm_int(b, 1),
                                 cosine ? nir_fneg(b, r) : complement,
                                 cosine ? complement : r);
         r = nir_bittest_select_agx3(b, quadrant, nir_imm_int(b, 2),
                                 nir_fneg(b, r), r);
      }
      nir_def *result = nir_fmul(b, nir_fsinc_agx3(b, r), r);
      nir_def *small = cosine ? nir_imm_float(b, 1) : x;

      /* sin x ~= x near zero, but handing the argument back verbatim performs
       * no arithmetic, so nothing ever flushes it.  Every fp32 denormal is far
       * below this threshold, so under DenormFlushToZero the passthrough
       * returns the very denormal the execution mode promises to flush --
       * dEQP-VK.spirv_assembly.instruction.compute.float_controls.fp32
       * .input_args.sin_denorm_flush_to_zero and its _vert/_frag twins.
       * Cosine is immune because its small-angle value is the constant 1.0.
       *
       * Canonicalize instead of selecting zero: sin x ~= x is a real
       * approximation for normal small arguments and must be kept.  NIR drops
       * the canonicalize when the mode is off, or when a consumer would flush
       * the value anyway, and lowers what is left to a multiply by one, which
       * the ALU flushes like any other arithmetic.
       *
       * This flushes at fp32.  A 16-bit sine is widened to fp32 above, so an
       * fp16 denormal argument is a normal fp32 value here and reappears when
       * the result is narrowed below; fp16 would need the flush after that
       * conversion.  shaderDenormFlushToZeroFloat16 is not advertised, so
       * nothing reaches that path today -- but it would bite the moment it is.
       */
      if (!cosine &&
          nir_is_denorm_flush_to_zero(
             b->shader->info.float_controls_execution_mode, bits))
         small = nir_fcanonicalize(b, small);

      result = nir_bcsel(b, nir_flt_imm(b, nir_fabs(b, x), 0x1p-12f),
                        small, result);
      /* Past the range the three-piece reduction can resolve there is no
       * quadrant left to recover and Apple's sequence answers zero.  An
       * infinite argument is not merely a large finite one: sin and cos of
       * an infinity are NaN, and a shader compiled without the NotInf or
       * NotNaN fast-math decorations is entitled to that NaN.  Separate the
       * two with an exact equality against infinity rather than letting the
       * range test fold them together.  A NaN argument never reaches this
       * select -- the comparison is false for it and the reduction above
       * already propagates it. */
      nir_def *magnitude = nir_fabs(b, x);
      nir_def *out_of_range =
         nir_flt(b, nir_imm_float(b, uif(0x4ac90fdb)), magnitude);
      nir_def *out_of_range_value =
         nir_bcsel(b, nir_feq(b, magnitude, nir_imm_float(b, uif(0x7f800000))),
                   nir_imm_float(b, uif(0x7fc00000)), nir_imm_float(b, 0));
      result = nir_bcsel(b, out_of_range, out_of_range_value, result);
      if (bits != 32)
         result = nir_f2fN(b, result, bits);
      b->fp_math_ctrl = saved_ctrl;
      return result;
   }

   nir_def *turns = nir_fmul_imm(b, x, M_1_PI * 0.5f);

   if (alu->op == nir_op_fcos)
      turns = nir_fadd_imm(b, turns, 0.25f);

   nir_def *quadrants = nir_fmul_imm(b, nir_ffract(b, turns), 4.0);
   return nir_fsin_agx(b, quadrants);
}

static bool
agx_lower_sincos(nir_shader *shader, unsigned native)
{
   return nir_shader_lower_instructions(shader, agx_lower_sincos_filter,
                                        agx_lower_sincos_impl,
                                        (void *)(uintptr_t)native);
}

static bool
agx_lower_front_face(struct nir_builder *b, nir_intrinsic_instr *intr,
                     UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_front_face)
      return false;

   nir_def *def = &intr->def;
   assert(def->bit_size == 1);

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_rewrite_uses(def, nir_inot(b, nir_load_back_face_agx(b, 1)));
   return true;
}

/* Lanes of a G17P SIMD group mostly agree on the kind of a light, so a rare
 * arm under a type test is skipped whole. Flattened, its rcp and rsq run every
 * loop iteration and every load it reads stays above the test. 24 is the best
 * of 12, 16, 24, 32 and 48 on TF3's light loop; smaller arms pay for a branch.
 */
#define AGX_G17P_EXPENSIVE_FLATTEN_LIMIT 24

/*
 * Standard NIR optimization loop. This is run in agx_preprocess_nir, then once
 * again at shader variant compile time. Unless there was a complex shader key,
 * the latter run should be almost a no-op.
 */
static void
agx_optimize_loop_nir(nir_shader *nir, bool m5, bool g17p)
{
   const nir_shader_compiler_options *saved_options = nir->options;
   nir_shader_compiler_options options;
   /* Native M5 ALU throughput benefits from exposing its 64-iteration
    * loops before scheduling. Keep earlier GPU generations unchanged. */
   if (m5) {
      options = *saved_options;
      options.max_unroll_iterations = 64;
      nir->options = &options;
   }
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 64,
         .expensive_alu_ok = true,
         .expensive_limit = g17p ? AGX_G17P_EXPENSIVE_FLATTEN_LIMIT : 0,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &peephole_select_options);
      NIR_PASS(progress, nir, nir_opt_phi_precision);
      /* The M5 conversion has D3D's saturating, NaN-to-zero semantics.
       * Fold before nir_opt_algebraic distributes a following add into the
       * clamping selects. */
      if (m5)
         NIR_PASS(progress, nir, agx_nir_fold_d3d_float_to_int);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
   nir->options = saved_options;
}

/* M5 multiplies two 32-bit registers into a 64-bit pair in one instruction
 * (imadd64, which the division lowering's umul_high already uses), so a
 * 32x32->64 product stays whole there and a 64x64 one is built on it.
 */
static bool
agx_nir_lower_imul64_filter(const nir_instr *instr, const void *data)
{
   const bool *native_2x32 = data;

   if (instr->type != nir_instr_type_alu)
      return false;

   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.bit_size != 64)
      return false;

   return alu->op == nir_op_imul ||
          (!*native_2x32 && (alu->op == nir_op_umul_2x32_64 ||
                             alu->op == nir_op_imul_2x32_64));
}

static nir_def *
agx_nir_umul_high32(nir_builder *b, nir_def *x, nir_def *y)
{
   nir_def *shift = nir_imm_int(b, 16);
   nir_def *mask = nir_imm_int(b, 0xffff);

   /* Hacker's Delight unsigned 16-bit partial-product expansion. */
   nir_def *xl = nir_iand(b, x, mask);
   nir_def *yl = nir_iand(b, y, mask);
   nir_def *xh = nir_ushr(b, x, shift);
   nir_def *yh = nir_ushr(b, y, shift);
   nir_def *lo = nir_imul(b, xl, yl);
   nir_def *m1 = nir_imul(b, xl, yh);
   nir_def *m2 = nir_imul(b, xh, yl);
   nir_def *hi = nir_imul(b, xh, yh);
   nir_def *t = nir_iadd(b, m2, nir_ushr(b, lo, shift));
   nir_def *w1 = nir_iadd(b, m1, nir_iand(b, t, mask));
   nir_def *w2 = nir_ushr(b, t, shift);
   return nir_iadd(b, hi, nir_iadd(b, nir_ushr(b, w1, shift), w2));
}

static nir_def *
agx_nir_lower_imul64_instr(nir_builder *b, nir_instr *instr, void *data)
{
   const bool *native_2x32 = data;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *y = nir_ssa_for_alu_src(b, alu, 1);
   if (alu->op == nir_op_umul_2x32_64 || alu->op == nir_op_imul_2x32_64) {
      nir_def *hi = agx_nir_umul_high32(b, x, y);
      if (alu->op == nir_op_imul_2x32_64) {
         /* Convert the unsigned high word to the signed product. */
         hi = nir_isub(b, hi, nir_iand(b, nir_ishr_imm(b, x, 31), y));
         hi = nir_isub(b, hi, nir_iand(b, nir_ishr_imm(b, y, 31), x));
      }
      return nir_pack_64_2x32_split(b, nir_imul(b, x, y), hi);
   }
   nir_def *xl = nir_unpack_64_2x32_split_x(b, x);
   nir_def *xh = nir_unpack_64_2x32_split_y(b, x);
   nir_def *yl = nir_unpack_64_2x32_split_x(b, y);
   nir_def *yh = nir_unpack_64_2x32_split_y(b, y);
   nir_def *cross = nir_iadd(b, nir_imul(b, xl, yh), nir_imul(b, xh, yl));

   if (*native_2x32) {
      nir_def *low = nir_umul_2x32_64(b, xl, yl);
      return nir_pack_64_2x32_split(
         b, nir_unpack_64_2x32_split_x(b, low),
         nir_iadd(b, nir_unpack_64_2x32_split_y(b, low), cross));
   }

   nir_def *hi = nir_iadd(b, agx_nir_umul_high32(b, xl, yl), cross);

   return nir_pack_64_2x32_split(b, nir_imul(b, xl, yl), hi);
}

static bool
agx_nir_lower_imul64(nir_shader *nir, bool native_2x32)
{
   return nir_shader_lower_instructions(
      nir, agx_nir_lower_imul64_filter, agx_nir_lower_imul64_instr,
      &native_2x32);
}

static bool
agx_nir_has_alu64(const nir_shader *nir, nir_op opcode)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            const nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == opcode && alu->def.bit_size == 64)
               return true;
         }
      }
   }

   return false;
}

struct agx_position_order_state {
   struct set *seen;
   struct util_dynarray todo;
};

static void
agx_position_order_add(struct agx_position_order_state *state, nir_def *def)
{
   if (_mesa_set_search(state->seen, def))
      return;
   _mesa_set_add(state->seen, def);
   util_dynarray_append(&state->todo, def);
}

static bool
agx_position_order_src(nir_src *src, void *data)
{
   agx_position_order_add(data, src->ssa);
   return true;
}

static void
agx_position_order_control(struct agx_position_order_state *state,
                           nir_block *block)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_if)
         agx_position_order_add(state, nir_cf_node_as_if(node)->condition.ssa);
   }
}

/* Preserve the source addition order of position, including shared operands.
 * A depth-only link must not reassociate its position differently merely
 * because a color link also uses an intermediate as a varying. Contracting
 * a multiply/add remains allowed; no other output is made invariant here. */
bool
agx_nir_preserve_position_order(nir_shader *nir)
{
   struct agx_position_order_state state = {
      .seen = _mesa_pointer_set_create(NULL),
   };
   util_dynarray_init(&state.todo, NULL);
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_output &&
                nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_POS) {
               agx_position_order_add(&state, intr->src[0].ssa);
               agx_position_order_control(&state, block);
            }
         }
      }
   }

   while (util_dynarray_num_elements(&state.todo, nir_def *)) {
      nir_def *def = util_dynarray_pop(&state.todo, nir_def *);
      nir_instr *instr = nir_def_instr(def);
      if (instr->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         unsigned flags = nir_op_valid_fp_math_ctrl(
            alu->op, nir_fp_no_reassoc | nir_fp_no_transform);
         progress |= (alu->fp_math_ctrl & flags) != flags;
         alu->fp_math_ctrl |= flags;
      } else if (instr->type == nir_instr_type_phi) {
         nir_foreach_phi_src(src, nir_instr_as_phi(instr))
            agx_position_order_control(&state, src->pred);
      }
      if (instr->block)
         agx_position_order_control(&state, instr->block);
      nir_foreach_src(instr, agx_position_order_src, &state);
   }
   util_dynarray_fini(&state.todo);
   _mesa_set_destroy(state.seen, NULL);

   /* Only ALU math-control flags changed: no instruction, def or block did. */
   nir_foreach_function_impl(impl, nir)
      nir_progress(progress, impl, nir_metadata_all);
   return progress;
}

bool
agx_mem_vectorize_cb(unsigned align_mul, unsigned align_offset,
                     unsigned bit_size, unsigned num_components,
                     int64_t hole_size, nir_intrinsic_instr *low,
                     nir_intrinsic_instr *high, void *data)
{
   if (hole_size > 0)
      return false;

   /* Must be aligned to the size of the load */
   unsigned align = nir_combined_align(align_mul, align_offset);
   if ((bit_size / 8) > align)
      return false;

   if (num_components > 4)
      return false;

   if (bit_size > 32)
      return false;

   return true;
}

static bool
set_speculate(nir_builder *b, nir_instr *instr, UNUSED void *_)
{
   if (instr->type == nir_instr_type_intrinsic &&
       nir_intrinsic_has_access(nir_instr_as_intrinsic(instr))) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_global_predicated_agx)
         return false;
      nir_intrinsic_set_access(
         intr, ACCESS_CAN_SPECULATE | nir_intrinsic_access(intr));
      return true;
   }

   if (instr->type == nir_instr_type_tex) {
      nir_instr_as_tex(instr)->can_speculate = true;
      return true;
   }

   return false;
}

static bool agx_g17p_discard_word(nir_shader *nir);
static bool agx_m5_discard_word(nir_shader *nir);

/* Drop a constant program left without instructions. */
static void
agx_remove_empty_preamble(nir_shader *nir)
{
   nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);
   if (!entrypoint->preamble)
      return;

   nir_foreach_block(block, entrypoint->preamble->impl) {
      if (!exec_list_is_empty(&block->instr_list))
         return;
   }
   exec_node_remove(&entrypoint->preamble->node);
   entrypoint->preamble = NULL;
}

/* Halves of the uniform file a preamble load/store covers. */
static unsigned
agx_preamble_halfs(const nir_def *def)
{
   return MAX2(def->bit_size, 16) * def->num_components / 16;
}

/*
 * A G17P vertex constant program runs once per vertex batch, not once per
 * draw, and costs a launch of its own there. One left with a few arithmetic
 * operations after the launch copies (a select on a copied flag, a mask)
 * saves the main program nothing it would not fold into the instructions
 * that use the result, so it is pure per-batch overhead (21 of 57 vertex
 * constant programs of a Subnautica frame: main +0..4 instructions without
 * them). Move such a program back into the main program. One that loads
 * from memory or does real work (a matrix product) stays.
 */
static void
agx3_inline_cheap_vertex_preamble(nir_shader *nir)
{
   /* Arithmetic the main program would repeat per vertex. */
   const unsigned max_ops = 4;

   nir_function_impl *main = nir_shader_get_entrypoint(nir);
   if (!main->preamble)
      return;

   nir_function_impl *pre = main->preamble->impl;
   if (exec_list_length(&pre->body) != 1)
      return;

   nir_block *block = nir_start_block(pre);
   /* Per half: 1 + the base of the store writing it, or 0. */
   uint16_t stored_halfs[512] = {0};
   const nir_def *stored[512] = {NULL};
   unsigned ops = 0;

   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_load_const:
      case nir_instr_type_undef:
         break;
      case nir_instr_type_alu: {
         nir_op op = nir_instr_as_alu(instr)->op;
         if (op != nir_op_mov && !nir_op_is_vec(op) &&
             op != nir_op_pack_64_2x32_split &&
             op != nir_op_unpack_64_2x32_split_x &&
             op != nir_op_unpack_64_2x32_split_y &&
             op != nir_op_pack_32_2x16_split &&
             op != nir_op_unpack_32_2x16_split_x &&
             op != nir_op_unpack_32_2x16_split_y && ++ops > max_ops)
            return;
         break;
      }
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_preamble) {
            if (nir_intrinsic_preamble_class(intr) !=
                nir_preamble_class_general)
               return;
            /* Reads its own results: not a straight copy into main. */
            const unsigned base = nir_intrinsic_base(intr);
            const unsigned size = agx_preamble_halfs(&intr->def);
            if (base + size > ARRAY_SIZE(stored_halfs))
               return;
            for (unsigned h = base; h < base + size; ++h) {
               if (stored_halfs[h])
                  return;
            }
         } else if (intr->intrinsic == nir_intrinsic_store_preamble) {
            if (nir_intrinsic_preamble_class(intr) !=
                nir_preamble_class_general)
               return;
            const unsigned base = nir_intrinsic_base(intr);
            const unsigned size = agx_preamble_halfs(intr->src[0].ssa);
            if (base + size > ARRAY_SIZE(stored_halfs))
               return;
            for (unsigned h = base; h < base + size; ++h) {
               if (stored_halfs[h])
                  return;
               stored_halfs[h] = base + 1;
            }
            stored[base] = intr->src[0].ssa;
         } else {
            return;
         }
         break;
      }
      default:
         return;
      }
   }

   /* Every main read of a result must be exactly one store's value. */
   nir_foreach_block(b, main) {
      nir_foreach_instr(instr, b) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
             intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
            const unsigned base = nir_intrinsic_desc_set(intr);
            for (unsigned h = base; h < MIN2(base + 4, 512); ++h) {
               if (stored_halfs[h])
                  return;
            }
            continue;
         }
         if (intr->intrinsic != nir_intrinsic_load_preamble)
            continue;
         const unsigned base = nir_intrinsic_base(intr);
         const unsigned size = agx_preamble_halfs(&intr->def);
         if (base + size > ARRAY_SIZE(stored_halfs))
            return;
         bool any = false;
         for (unsigned h = base; h < base + size; ++h)
            any |= stored_halfs[h] != 0;
         if (any && (stored_halfs[base] != base + 1 ||
                     stored[base]->bit_size != intr->def.bit_size ||
                     stored[base]->num_components !=
                        intr->def.num_components))
            return;
      }
   }

   /* Recompute the results at the top of main, then read them there. */
   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   nir_def *value[512] = {NULL};
   nir_builder b = nir_builder_at(nir_before_impl(main));

   nir_foreach_instr(instr, block) {
      if (instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic ==
             nir_intrinsic_store_preamble) {
         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         struct hash_entry *e =
            _mesa_hash_table_search(remap, store->src[0].ssa);
         value[nir_intrinsic_base(store)] = e ? e->data : store->src[0].ssa;
         continue;
      }

      nir_instr *clone = nir_instr_clone_deep(nir, instr, remap);
      nir_builder_instr_insert(&b, clone);
   }

   nir_foreach_block(blk, main) {
      nir_foreach_instr_safe(instr, blk) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
         if (load->intrinsic != nir_intrinsic_load_preamble)
            continue;
         const unsigned base = nir_intrinsic_base(load);
         nir_def *v = value[base];
         if (!v || !stored_halfs[base])
            continue;
         assert(v->bit_size == load->def.bit_size &&
                v->num_components == load->def.num_components);
         nir_def_replace(&load->def, v);
      }
   }

   _mesa_hash_table_destroy(remap, NULL);
   exec_node_remove(&main->preamble->node);
   main->preamble = NULL;
   nir_progress(true, main, nir_metadata_none);
}

/* The hardware takes a derivative from the neighbouring lanes' registers, so
 * every lane of the quad must have computed the operand. A source loaded before
 * a branch but sunk into it is computed only by the lanes that branch, and a
 * derivative inside that branch then reads garbage from the others. Explicit
 * derivatives in control flow are rare, so keep sinking everywhere else. */
static bool
agx_has_derivative_in_control_flow(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent == &impl->cf_node)
            continue;

         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_ddx:
            case nir_intrinsic_ddx_fine:
            case nir_intrinsic_ddx_coarse:
            case nir_intrinsic_ddy:
            case nir_intrinsic_ddy_fine:
            case nir_intrinsic_ddy_coarse:
               return true;
            default:
               break;
            }
         }
      }
   }
   return false;
}


static bool agx3_nir_lower_uniform_dma(nir_shader *nir,
                                       const struct agx_shader_key *key,
                                       struct agx_shader_info *info);

/* G18 has no admitted native quad or integer min/max/bitwise prefix encoding.
 * Use the common activity-aware subgroup algorithms, then let the normal AGX
 * pass lower generated shuffles and ballots. Whole-subgroup native operations
 * stay intact. */
static bool
agx_g18_subgroup_filter(const nir_intrinsic_instr *intr, const void *data)
{
   (void)data;
   switch (intr->intrinsic) {
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
      return true;
   case nir_intrinsic_reduce: {
      unsigned cluster = nir_intrinsic_cluster_size(intr);
      return cluster > 0 && cluster < 32;
   }
   case nir_intrinsic_inclusive_scan:
   case nir_intrinsic_exclusive_scan:
      break;
   default:
      return false;
   }

   switch (nir_intrinsic_reduction_op(intr)) {
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
      return true;
   default:
      return false;
   }
}

static bool
agx_nir_lower_g18_subgroups(nir_shader *nir)
{
   const nir_lower_subgroups_options options = {
      .filter = agx_g18_subgroup_filter,
      .lower_reduce = true,
      .lower_quad = true,
      .lower_to_scalar = true,
      .ballot_bit_size = 32,
      /* Keep NIR's canonical ballot shape until the normal AGX pass. */
      .ballot_components = 4,
      .subgroup_size = 32,
   };
   bool progress = agx_nir_flush_subgroup_minmax(nir);
   while (nir_lower_subgroups(nir, &options))
      progress = true;
   return progress;
}

static void
agx_optimize_nir(nir_shader *nir, bool soft_fault, bool uniform_atomics,
                 bool byte_indices, bool discard_word, bool g17p, bool m5,
                 bool g18_scans, bool checked_offsets,
                 uint16_t *preamble_size, uint8_t *ts_count, uint8_t *ss_count,
                 const struct agx_shader_key *dma_key,
                 struct agx_shader_info *dma_info)
{
   /* This runs only once up front since other optimizations don't affect it */
   NIR_PASS(_, nir, nir_opt_shrink_stores, true);

   agx_optimize_loop_nir(nir, m5, g17p);

   /* Loops are unrolled now: a subgroup XOR shuffle whose mask is still not
    * a constant takes the general lowering (agx_nir_lower_subgroups.c).
    */
   if (m5 && !g18_scans) {
      bool lowered = false;
      NIR_PASS(lowered, nir, agx_nir_lower_dynamic_shuffle_xor);
      if (lowered)
         agx_optimize_loop_nir(nir, m5, g17p);
   }

   /* If soft fault is enabled, we can freely speculate everything. That lets us
    * peephole select and form preambles more aggressively.
    */
   if (soft_fault) {
      NIR_PASS(_, nir, nir_shader_instructions_pass, set_speculate,
               nir_metadata_control_flow, NULL);
   }

   /* Peephole select again after setting the speculate flag but before
    * vectorizing. This cleans up short-circuit loads in unrolled loops.
    *
    * XXX: Set indirect_load_ok once we can investigate CTS flakes.
    */
   nir_opt_peephole_select_options peephole_select_options = {
      .limit = 64,
      .expensive_alu_ok = true,
      .expensive_limit = g17p ? AGX_G17P_EXPENSIVE_FLATTEN_LIMIT : 0,
   };
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);

   /* Small uniform ifs whose arms load from memory, which peephole select
    * keeps: on M5 an if costs five control-flow instructions and a wait for
    * every load in flight. Flattening one can expose another around it.
    * Not on G18G, which compiles as AGX3C with emit_agx3d (g18_scans). */
   if (m5 && !g18_scans) {
      bool flattened;
      do {
         flattened = false;
         NIR_PASS(flattened, nir, agx_nir_opt_flatten_loads, 64, 4);
         NIR_PASS(flattened, nir, nir_opt_peephole_select,
                  &peephole_select_options);
      } while (flattened);
      NIR_PASS(_, nir, nir_opt_algebraic);
      NIR_PASS(_, nir, nir_opt_cse);
      NIR_PASS(_, nir, nir_opt_dce);
   }

   NIR_PASS(_, nir, nir_opt_load_store_vectorize,
            &(const nir_load_store_vectorize_options){
               .modes = nir_var_mem_global | nir_var_mem_constant |
                        nir_var_shader_temp,
               .callback = agx_mem_vectorize_cb,
            });
   NIR_PASS(_, nir, nir_lower_pack);
   NIR_PASS(_, nir, nir_opt_algebraic);

   nir_divergence_analysis(nir);
   bool progress = false;

   static const nir_lower_subgroups_options subgroups_options = {
      .ballot_bit_size = 32,
      .ballot_components = 1,
      .lower_elect = true,
      .lower_subgroup_masks = true,
   };

   /*
    * Rewrites a uniform-address atomic into "one lane of the subgroup performs
    * the whole operation", which needs ACTIVE_THREAD_INDEX_IN_SUBGROUP and
    * TOTAL_ACTIVE_THREADS_IN_SUBGROUP. On AGX3 the caller enables it only in
    * the stages where the packer admits those (SR 34/36). Skipping it keeps
    * atomics correct, at one atomic per lane instead of one per subgroup.
    */
   NIR_PASS(_, nir, agx_nir_lower_atomics, byte_indices);

   if (uniform_atomics)
      NIR_PASS(progress, nir, nir_opt_uniform_atomics, true);

   /* Lower addressing modes. The sooner we do this, the sooner we get rid of
    * amul/aadd instructions and can let nir_opt_algebraic do its job. But we
    * want to vectorize first since nir_opt_load_store_vectorize doesn't know
    * how to handle our loads. Likewise for uniform atomic optimization.
    */
   NIR_PASS(_, nir, agx_nir_lower_address, byte_indices, m5,
            checked_offsets);

   /* Now that loads carry an index, a select between two of them can choose
    * the index instead of the loaded data. Run it to a fixed point with the
    * cleanup below, so a chain of candidate selects collapses to one load.
    * poly's geometry rasterization copy loaded every candidate vertex per
    * varying, which overflowed the AGX3B register file and spilled.
    */
   NIR_PASS(_, nir, agx_nir_opt_select_load);
   NIR_PASS(_, nir, nir_opt_cse);
   NIR_PASS(_, nir, nir_opt_dce);

   NIR_PASS(progress, nir, nir_opt_uniform_subgroup, &subgroups_options);
   if (g18_scans)
      NIR_PASS(progress, nir, agx_nir_lower_g18_subgroups);
   if (progress) {
      if (byte_indices)
         NIR_PASS(_, nir, agx_nir_lower_subgroups_g17p);
      else if (m5)
         NIR_PASS(_, nir, agx_nir_lower_subgroups_m5, !g18_scans, !g18_scans);
      else
         NIR_PASS(_, nir, agx_nir_lower_subgroups);
   }

   /* The above create operations that need lowering/optimizing */
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   progress = false;

   /* If address lowering made progress, clean up before forming preambles.
    * Otherwise the optimized preambles might just be constants! Do it before
    * lowering int64 too, to avoid lowering constant int64 arithmetic.
    */
   if (progress) {
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_dce);
   }

   /* Only lower int64 after optimizing address arithmetic, so that u2u64/i2i64
    * conversions remain.
    */
   progress = false;
   if (byte_indices)
      NIR_PASS(progress, nir, agx_nir_scalarize_int64);
   /* G17P and M5: normalized 64-bit division instead of nir_lower_int64's
    * fully unrolled 64-step division (agx_nir_lower_divmod64.c). M5 keeps
    * the loop's 64-bit add and subtract native. */
   if (g17p || m5)
      NIR_PASS(progress, nir, agx_nir_lower_divmod64, m5);
   NIR_PASS(progress, nir, nir_lower_int64);

   /* If we lowered actual int64 arithmetic (not folded into the address
    * calculations), then clean up after the lowering.
    */
   if (progress) {
      do {
         progress = false;

         NIR_PASS(progress, nir, nir_opt_algebraic);
         NIR_PASS(progress, nir, nir_opt_constant_folding);
         NIR_PASS(progress, nir, nir_opt_dce);
      } while (progress);
   }

   /* Reassociate before forming preambles because it makes preambles more
    * effective. Clean up after. On M5, also order the loop-invariant terms of
    * a chain in a loop first, so accumulations of per-iteration constants
    * (robustness loops add the same size queries every iteration) are summed
    * once in the preamble instead of once per term and iteration.
    */
   nir_opt_reassociate_loop(
      nir, nir_reassociate_scalar_math | nir_reassociate_cse_heuristic |
              (m5 ? nir_reassociate_loop_invariant : 0));

   /* Lower fmin/fmax before optimizing preambles so we can see across uniform
    * expressions. Do it after nir_opt_reassociate because nir_opt_reassociate
    * should work on unlowered fmin/fmax.
    */
   NIR_PASS(_, nir, agx_nir_lower_fminmax, g17p || m5);

   /* AGX_G17P_DISCARD_WORD: reduce one-sample coverage-update sources to bit
    * 0 after the algebraic loops and right before preamble formation, so the
    * uniform `tests_zs & 1` is hoisted. nir_opt_algebraic would otherwise
    * refactor `b2i(c) | (u & 1)` back into `(b2i(c) | u) & 1` (an OR and an
    * AND per fragment). */
   if (discard_word) {
      bool word = false;
      /* G18G's one-sample MOVMSK takes the G17P-reduced word (the backend
       * keeps it on agx_g18_sample_mask_word), not the M5 form. */
      if (m5 && !g18_scans)
         NIR_PASS(word, nir, agx_m5_discard_word);
      else
         NIR_PASS(word, nir, agx_g17p_discard_word);
      if (word) {
         NIR_PASS(_, nir, nir_opt_constant_folding);
         NIR_PASS(_, nir, nir_opt_dce);
      }
   }

   /* Force preamble generation for internal shaders since hk's meta copies
    * depend on it for correctness with imgwblk intrinsics.
    */
   if (preamble_size &&
       (!(agx_compiler_debug & AGX_DBG_NOPREAMBLE) || nir->info.internal)) {
      /* AGX3 (byte_indices: the AGX3B encoding) has no texture/sampler
       * state-store encoding: its descriptor lowering uses immediate native
       * state slots, so a bindless handle hoisted into the image/sampler
       * preamble class becomes a tex_state_store/sampler_state_store that
       * the AGX3 packer refuses (an internal meta fragment shader storing
       * through a bindless image failed vkCreateGraphicsPipelines). Keep
       * bindless handles in the main shader there; nothing that compiles
       * today hoists either class. */
      const bool state_promotion = !(byte_indices || m5);
      unsigned sizes[] = {
         *preamble_size,
         (ts_count && state_promotion) ? *ts_count : 1000 /* large finite */,
         (ss_count && state_promotion) ? *ss_count : 1000 /* large finite */,
      };

      /* Don't clobber txf sampler */
      if (sizes[2] == 0)
         sizes[2]++;

      NIR_PASS(_, nir, agx_nir_opt_preamble, sizes, byte_indices || m5, m5,
               &peephole_select_options);

      *preamble_size = sizes[0];

      /* M5: check the constant program's constant-offset buffer reads past
       * 255 bytes in words, so their ends stay immediates
       * (agx_nir_lower_word_bounds.c). The main program keeps its checks.
       * A uniform DMA launch drops the byte checks of the reads it copies by
       * matching their byte form, so leave those programs alone. */
      if (m5 && !(dma_key && dma_key->g17p_uniform_dma)) {
         bool words = false;
         NIR_PASS(words, nir, agx_nir_lower_word_bounds);
         if (words) {
            NIR_PASS(_, nir, nir_opt_cse);
            NIR_PASS(_, nir, nir_opt_dce);
         }
      }

      /* M5: let the main program's robust reads of one structure share their
       * end computation (agx_nir_share_bounds.c). */
      if (m5) {
         bool shared = false;
         NIR_PASS(shared, nir, agx_nir_share_bounds);
         if (shared) {
            NIR_PASS(_, nir, nir_opt_copy_prop);
            NIR_PASS(_, nir, nir_opt_dce);
         }
      }

      /* Copy what the launch can copy into the uniforms the constant
       * program leaves free, which then computes on the copies instead of
       * loading. Copying first would take uniforms the constant program's
       * results need. */
      if (dma_key && dma_key->g17p_uniform_dma &&
          agx3_nir_lower_uniform_dma(nir, dma_key, dma_info)) {
         NIR_PASS(_, nir, nir_opt_dce);
         agx_remove_empty_preamble(nir);
      }

      if (dma_key && dma_key->g17p_uniform_dma &&
          nir->info.stage == MESA_SHADER_VERTEX) {
         agx3_inline_cheap_vertex_preamble(nir);
         NIR_PASS(_, nir, nir_opt_dce);
      }

      if (ts_count && state_promotion)
         *ts_count = sizes[1];

      /* if something other than the txf sampler is written... */
      if (ss_count && state_promotion && sizes[2] > 1)
         *ss_count = sizes[2];
   }

   /* Forming preambles may dramatically reduce the instruction count
    * in certain blocks, causing some if-else statements to become
    * trivial. We want to peephole select those, given that control flow
    * prediction instructions are costly.
    *
    * We need to lower int64 again to deal with the resulting 64-bit csels.
    */
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);
   if (byte_indices)
      NIR_PASS(_, nir, agx_nir_scalarize_int64);
   NIR_PASS(_, nir, nir_lower_int64);

   /* Constant-program flags, now that their trivial branches are selects. */
   if (byte_indices && !m5)
      NIR_PASS(_, nir, agx_nir_export_select_flags_as_halves, true);

   /* After the selects of trivial if-else statements exist. */
   if (m5) {
      bool folded = false;
      NIR_PASS(folded, nir, agx_nir_m5_split_or_selects);
      if (folded) {
         NIR_PASS(_, nir, nir_opt_copy_prop);
         NIR_PASS(_, nir, nir_opt_dce);
      }
   }

   /* We need to lower fmin/fmax again after nir_opt_algebraic_late due to f2fmp
    * wackiness. This is usually a no-op but is required for correctness in
    * GLES.
    */
   NIR_PASS(_, nir, nir_opt_algebraic_late);
   NIR_PASS(_, nir, agx_nir_lower_fminmax, g17p || m5);

   /* The AGX3 byte-address backend cannot encode 64-bit shifted adds.
    * Lower remaining multiplies before fusion turns x * 3 into one; the
    * normal late multiply pass cannot recognize fused backend opcodes. */
   if (byte_indices)
      NIR_PASS(_, nir, agx_nir_lower_imul64, false);

   /* Before the shift fusions below rewrite the byte extracts it matches. */
   /* M5 only: G18 legalizes byte conversions through its halfword forms
    * (agx_lower_agx3_half_converts). */
   if (m5 && !g18_scans) {
      NIR_PASS(_, nir, agx_nir_fold_byte_to_float);
      NIR_PASS(_, nir, agx_nir_fold_ubyte_to_float);
   }

   /* Fuse add/sub/multiplies/shifts after running opt_algebraic_late to fuse
    * isub but before shifts are lowered.
    */
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, agx_nir_fuse_algebraic_late);
      if (m5)
         NIR_PASS(progress, nir, agx_nir_fuse_doubled_add);
   } while (progress);

   /* Before optimizing bounds checks, we need to clean up and index defs so
    * optimize_bounds does the right thing.
    */
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);

   nir_foreach_function_impl(impl, nir) {
      nir_index_ssa_defs(impl);
   }

   /* Do remaining lowering late, since this inserts &s for shifts so we want to
    * do it after fusing constant shifts. Constant folding will clean up.
    */
   /* Must precede agx_nir_lower_algebraic_late, which rewrites extract_u16
    * into u2u32(unpack_32_2x16_split_y) (the pass also matches that form). */
   NIR_PASS(_, nir, agx_nir_fold_extract16_bfe);
   NIR_PASS(_, nir, agx_nir_lower_algebraic_late);
   if (g18_scans)
      NIR_PASS(_, nir, agx_nir_avoid_square_overflow);
   NIR_PASS(_, nir, agx_nir_fold_pixel_coord);
   NIR_PASS(_, nir, agx_nir_fuse_selects);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_combine_barriers, NULL, NULL);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_cse);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);

   /* Cleanup optimizations */
   nir_move_options move_all = nir_move_const_undef | nir_move_load_ubo |
                               nir_move_load_input | nir_move_load_frag_coord |
                               nir_move_comparisons | nir_move_copies |
                               nir_move_load_ssbo | nir_move_alu;
   if (agx_has_derivative_in_control_flow(nir))
      move_all &= ~(nir_move_load_input | nir_move_load_frag_coord |
                    nir_move_alu);

   /* Sinking a G17P buffer load into the arm that reads it saves the load when
    * the arm is skipped. Sink leaves it at the top of that block; moving it on
    * to its use would make each load wait alone instead of overlapping.
    */
   NIR_PASS(_, nir, nir_opt_sink,
            g17p ? move_all : move_all & ~nir_move_load_ssbo);
   NIR_PASS(_, nir, nir_opt_move, move_all & ~nir_move_load_ssbo);
   NIR_PASS(progress, nir, nir_lower_all_phis_to_scalar);

   /* After lowering phis to scalar, we must copy prop/DCE for correctness. RA
    * can't deal with dead phis.
    */
   do {
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      progress = false;
   } while (progress);
}

/*
 * Varyings that are used as texture coordinates should be kept at fp32, because
 * fp16 does not have enough precision for large textures. It's technically
 * conformant not to, but every app gets this wrong.
 */
static bool
gather_texcoords(nir_builder *b, nir_tex_instr *tex, void *data)
{
   nir_def *coord = nir_get_tex_src(tex, nir_tex_src_coord);
   if (!coord)
      return false;

   nir_scalar x = nir_scalar_resolved(coord, 0);
   nir_scalar y = nir_scalar_resolved(coord, 1);
   nir_intrinsic_instr *intr = nir_scalar_as_intrinsic(x);

   if (x.def == y.def && intr &&
       intr->intrinsic == nir_intrinsic_load_interpolated_input) {

      uint64_t *mask = data;
      *mask |= BITFIELD64_BIT(nir_intrinsic_io_semantics(intr).location);
   }

   return false;
}

/*
 * Build a bit mask of varyings (by location) that are used as texture
 * coordinates. This information is needed by lower_mediump_io.
 */
uint64_t
agx_gather_texcoords(nir_shader *nir)
{
   assert(nir->info.stage == MESA_SHADER_FRAGMENT);

   uint64_t mask = 0;
   nir_shader_tex_pass(nir, gather_texcoords, nir_metadata_all, &mask);
   return mask;
}

static nir_mem_access_size_align
mem_access_size_align_cb(nir_intrinsic_op intrin, uint8_t bytes,
                         uint8_t bit_size, uint32_t align,
                         uint32_t align_offset, bool offset_is_const,
                         enum gl_access_qualifier access, const void *cb_data)
{
   align = nir_combined_align(align, align_offset);

   assert(util_is_power_of_two_nonzero(align));

   if ((bytes & 1) || (align == 1))
      bit_size = 8;
   else if ((bytes & 2) || (align == 2))
      bit_size = 16;
   else if (bit_size >= 32)
      bit_size = 32;

   return (nir_mem_access_size_align){
      .num_components = MIN2(bytes / (bit_size / 8), 4),
      .bit_size = bit_size,
      .align = bit_size / 8,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

static unsigned
lower_bit_size_callback(const nir_instr *instr, UNUSED void *_)
{
   if (instr->type == nir_instr_type_intrinsic) {
      /* Handle small subgroup ops */
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_reduce:
      case nir_intrinsic_exclusive_scan:
      case nir_intrinsic_inclusive_scan:
         /* The identity for iand doesn't work for lowered 1-bit booleans, so
          * lower that explicitly.
          */
         if (nir_intrinsic_reduction_op(intr) == nir_op_iand &&
             intr->def.bit_size == 1)
            return 16;

         /* In general, we have 16-bit ops instead of 8-bit, so lower those. */
         return intr->def.bit_size == 8 ? 16 : 0;
      default:
         return 0;
      }
   } else if (instr->type == nir_instr_type_alu) {
      /* Lower 8-bit ALU to 16-bit. We check the destination, as we do not want
       * to lower conversions from 8-bit to larger types. Those conversions get
       * implemented natively.
       */
      nir_alu_instr *alu = nir_instr_as_alu(instr);

      switch (alu->op) {
      case nir_op_bitfield_reverse:
         return alu->def.bit_size < 32 ? 32 : 0;
      default:
         break;
      }

      if (alu->def.bit_size == 8 && !is_conversion_to_8bit(alu->op))
         return 16;
      else if (alu->def.bit_size == 1 && alu->src[0].src.ssa->bit_size == 8)
         return 16 /* comparisons */;
   }

   return 0;
}

static bool
lower_load_from_texture_handle(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_from_texture_handle_agx)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def *base = nir_load_preamble(b, 1, 64, nir_intrinsic_desc_set(handle));
   nir_def *offset = nir_u2u64(b, handle->src[0].ssa);

   nir_def_rewrite_uses(&intr->def, nir_iadd(b, base, offset));
   return true;
}

static bool
lower_sink_address(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ro_sink_address_poly:
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
      return true;
   case nir_intrinsic_ro_to_rw_poly: {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(
         &intr->def,
         nir_bcsel(b, nir_ieq_imm(b, intr->src[0].ssa, AGX_ZERO_PAGE_ADDRESS),
                   nir_imm_int64(b, AGX_SCRATCH_PAGE_ADDRESS),
                   intr->src[0].ssa));
      return true;
   }
   default:
      break;
   }

   return false;
}

static void
agx_remove_unreachable_block(agx_block *block)
{
   /* Delete the edges */
   agx_foreach_successor(block, succ) {
      unsigned block_idx = agx_predecessor_index(succ, block);

      /* Remove the corresponding predecessor from the successor */
      struct util_dynarray *blocks = &succ->predecessors;
      int remaining = agx_num_predecessors(succ) - (block_idx + 1);
      assert(remaining >= 0);

      memcpy(util_dynarray_element(blocks, agx_block *, block_idx),
             util_dynarray_element(blocks, agx_block *, block_idx + 1),
             remaining * sizeof(agx_block *));
      blocks->size -= sizeof(agx_block *);

      /* Remove the corresponding source from the phis */
      agx_foreach_phi_in_block(succ, phi) {
         assert(block_idx + 1 <= phi->nr_srcs);

         memcpy(phi->src + block_idx, phi->src + block_idx + 1,
                (phi->nr_srcs - (block_idx + 1)) * sizeof(phi->src[0]));

         phi->nr_srcs--;

         /* This might cause phis to become trivial. Lower 1-source phis to
          * moves and let copyprop take it from here.
          */
         if (phi->nr_srcs == 1) {
            phi->op = AGX_OPCODE_MOV;
         }
      }
   }

   /* Remove the successor from the predecessor. */
   block->successors[0] = NULL;
   block->successors[1] = NULL;

   /* Note: we do not remove the block itself, although it is now fully orphaned
    * in the control flow graph. We still need it in source order if it has any
    * pop_exec instructions, for a loop continue block.
    *
    * TODO: Is there a better way to handle this?
    *
    * Affects: dEQP-VK.graphicsfuzz.cov-matching-if-always-true-inside-loop
    */
}

/*
 * NIR sometimes contains unreachable blocks (e.g. due to infinite loops). These
 * blocks have no predecessors, but do have successors and can contribute to
 * phis. They are dead and do not need to be here. Further, they violate the IR
 * invariant:
 *
 *    Live-in sources are live-out in all predecessors.
 *
 * ...which RA depends on when handling live range splits. The simplest solution
 * is to simply delete these dead blocks. Fortunately, because they are
 * unreachable, this does not have any ill effects. Notably, this cannot
 * introduce critical edges.
 *
 * Deleting a block may cause a successor to become unreachable, so we use a
 * fixed-point algorithm to converge.
 */
static void
agx_remove_unreachable_blocks(agx_context *ctx)
{
   agx_block *start = agx_start_block(ctx);
   bool progress;

   do {
      progress = false;

      agx_foreach_block_safe(ctx, pred) {
         if (pred != start && agx_num_predecessors(pred) == 0 &&
             agx_num_successors(pred) > 0) {

            agx_remove_unreachable_block(pred);
            progress = true;
         }
      }
   } while (progress);
}

static bool
agx_should_dump(nir_shader *nir, unsigned agx_dbg_bit)
{
   return (agx_compiler_debug & agx_dbg_bit) &&
          !(nir->info.internal && !(agx_compiler_debug & AGX_DBG_INTERNAL));
}

#define AGX_PASS(shader, pass, ...)                                            \
   do {                                                                        \
      pass(shader, ##__VA_ARGS__);                                             \
      agx_validate(ctx, #pass);                                                \
   } while (0)

/* G17P constant programs export whole 32-bit uniform words. Track both halves
 * of each word through disjoint exports, in either order, so separated scalar
 * results can use the existing COPY encoding. Only SSA/immediate values may
 * be delayed; all observable uniform reads, unknown writes and barriers end
 * the candidate range. Nothing moves across a basic-block boundary. */
bool
agx3_coalesce_uniform_half_stores(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       !ctx->is_preamble)
      return false;

   /* Three-half exports may share a 32-bit word with another live export.
    * Expose their individual halves to the existing ordered coalescer, rather
    * than padding across a neighbour. Keep an aligned vec3 with an unwritten
    * fourth half intact for the existing whole-vector padding path. */
   bool written[512] = {false};
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_UNIFORM_STORE || I->nr_srcs != 2 ||
          I->src[1].type != AGX_INDEX_IMMEDIATE || I->mask > 15)
         continue;
      const unsigned base = I->src[1].value;
      const unsigned stride = agx_size_align_16(I->src[0].size);
      for (unsigned c = 0; c < 4; ++c) {
         if (!(I->mask & BITFIELD_BIT(c)) || base >= 512 ||
             (c + 1) * stride > 512 - base)
            continue;
         for (unsigned h = 0; h < stride; ++h)
            written[base + c * stride + h] = true;
      }
   }

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_UNIFORM_STORE || I->nr_srcs != 2 ||
          I->mask != 7 || I->src[1].type != AGX_INDEX_IMMEDIATE ||
          I->src[1].abs || I->src[1].neg || I->src[1].memory)
         continue;
      const agx_index value = I->src[0];
      const unsigned base = I->src[1].value;
      if (value.type != AGX_INDEX_NORMAL || value.size != AGX_SIZE_16 ||
          agx_channels(value) != 3 || value.memory || value.abs || value.neg ||
          base > 509 || (!(base & 1) && !written[base + 3]))
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_instr *split = agx_split(&b, 3, value);
      for (unsigned h = 0; h < 3; ++h) {
         split->dest[h] = agx_temp(ctx, AGX_SIZE_16);
         agx_uniform_store(&b, split->dest[h], agx_immediate(base + h), 1);
      }
      agx_remove_instruction(I);
      progress = true;
   }
   agx_foreach_block(ctx, block) {
      agx_instr *pending[256] = {0};

      agx_foreach_instr_in_block_safe(block, I) {
         /* A fixed-file read observes preceding exports at this point.
          * LOAD_CPROG_AGX3 instead reads immutable imported TPS storage,
          * which is not changed by any uniform export in this program. */
         agx_foreach_src(I, s) {
            if (I->src[s].type == AGX_INDEX_UNIFORM)
               memset(pending, 0, sizeof(pending));
         }
         agx_foreach_dest(I, d) {
            if (I->dest[d].type == AGX_INDEX_UNIFORM)
               memset(pending, 0, sizeof(pending));
         }

         if (I->op == AGX_OPCODE_UNIFORM_STORE) {
            if (I->nr_srcs != 2 || !I->mask || I->mask > 15 ||
                I->src[1].type != AGX_INDEX_IMMEDIATE || I->src[1].abs ||
                I->src[1].neg || I->src[1].memory) {
               memset(pending, 0, sizeof(pending));
               continue;
            }

            const unsigned base = I->src[1].value;
            const agx_index value = I->src[0];
            const unsigned stride = agx_size_align_16(value.size);
            const bool scalar_half =
               I->mask == 1 && value.size == AGX_SIZE_16 &&
               agx_channels(value) == 1 && !value.abs && !value.neg &&
               !value.memory &&
               (value.type == AGX_INDEX_NORMAL ||
                value.type == AGX_INDEX_IMMEDIATE) && base < 512;

            if (scalar_half) {
               agx_instr *other = pending[base / 2];
               if (other && (other->src[1].value ^ base) == 1) {
                  agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
                  agx_index halves[2];
                  halves[base & 1] = value;
                  halves[(base & 1) ^ 1] = other->src[0];
                  /* Stores accept 16-bit immediates, but COLLECT accepts
                   * only 8-bit ones. Keep wide half constants in SSA. */
                  for (unsigned h = 0; h < 2; ++h) {
                     if (halves[h].type == AGX_INDEX_IMMEDIATE &&
                         halves[h].value >= 256)
                        halves[h] = agx_mov_imm(&b, 16, halves[h].value);
                  }
                  agx_index pair = agx_vec_temp(ctx, AGX_SIZE_32, 1);
                  agx_emit_collect_to(&b, pair, 2, halves);
                  agx_uniform_store(&b, pair, agx_immediate(base & ~1u), 1);
                  agx_remove_instruction(other);
                  agx_remove_instruction(I);
                  pending[base / 2] = NULL;
                  progress = true;
               } else {
                  /* A repeated write replaces only this candidate. Its
                   * preceding instruction remains unless another pass
                   * proves it dead. */
                  pending[base / 2] = I;
               }
               continue;
            }

            /* A non-candidate export invalidates every pair it overlaps.
             * Disjoint exports leave other pending SSA values available. */
            for (unsigned c = 0; c < 4; ++c) {
               if (!(I->mask & BITFIELD_BIT(c)))
                  continue;
               if (base >= 512 || (c + 1) * stride > 512 - base) {
                  memset(pending, 0, sizeof(pending));
                  break;
               }
               for (unsigned h = 0; h < stride; ++h)
                  pending[(base + c * stride + h) / 2] = NULL;
            }
         } else if (I->op == AGX_OPCODE_UNIFORM_FADD_AGX3 ||
                    I->op == AGX_OPCODE_UNIFORM_FMUL_AGX3 ||
                    I->op == AGX_OPCODE_UNIFORM_FFMA_AGX3) {
            if (I->imm > 510) {
               memset(pending, 0, sizeof(pending));
            } else {
               pending[I->imm / 2] = NULL;
               pending[(I->imm + 1) / 2] = NULL;
            }
         } else if (!agx_opcodes_info[I->op].can_eliminate ||
                    agx_opcodes_info[I->op].schedule_class ==
                       AGX_SCHEDULE_CLASS_BARRIER ||
                    agx_opcodes_info[I->op].schedule_class ==
                       AGX_SCHEDULE_CLASS_INVALID) {
            memset(pending, 0, sizeof(pending));
         }
      }
   }

   return progress;
}

/* Native G17P fragment programs issue independent LDDEVs sharing a register
 * base in increasing address order. The row-major UBO path is the minimal
 * counterexample: NIR presents offsets 16 then 0, both become ready together,
 * and the resulting descending pair returns intermittent stale data while the
 * otherwise identical ascending column-major path is stable. Sort only an
 * adjacent run of non-coherent loads with identical operation and base; SSA
 * keeps their consumers attached to the correct result and no serialization
 * is introduced. */
static bool
agx3_order_fragment_device_loads(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       ctx->stage != MESA_SHADER_FRAGMENT)
      return false;

   bool progress = false;
   agx_foreach_block(ctx, block) {
      agx_instr *run[16];
      unsigned count = 0;

      agx_foreach_instr_in_block(block, I) {
         const bool candidate =
            I->op == AGX_OPCODE_DEVICE_LOAD && !I->coherent &&
            I->nr_dests == 1 && I->nr_srcs == 2 &&
            I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].abs &&
            !I->src[1].neg && !I->src[1].memory;
         const bool joins =
            candidate && count > 0 && count < ARRAY_SIZE(run) &&
            agx_is_equiv(I->src[0], run[0]->src[0]) &&
            I->format == run[0]->format && I->mask == run[0]->mask &&
            I->memory_index_is_bytes == run[0]->memory_index_is_bytes;

         /* SPLIT only describes the preceding SSA result and emits no machine
          * instruction. It may sit between the loads in an otherwise adjacent
          * hardware run. */
         if (!candidate && I->op == AGX_OPCODE_SPLIT)
            continue;

         if (!candidate || (count > 0 && !joins)) {
            for (unsigned i = 1; i < count; ++i) {
               for (unsigned j = i; j > 0 &&
                    ((uint64_t)run[j - 1]->src[1].value << run[j - 1]->shift) >
                    ((uint64_t)run[j]->src[1].value << run[j]->shift); --j) {
                  agx_instr *lower = run[j];
                  list_del(&lower->link);
                  list_addtail(&lower->link, &run[j - 1]->link);
                  run[j] = run[j - 1];
                  run[j - 1] = lower;
                  progress = true;
               }
            }
            count = 0;
         }

         if (candidate)
            run[count++] = I;
      }

      for (unsigned i = 1; i < count; ++i) {
         for (unsigned j = i; j > 0 &&
              ((uint64_t)run[j - 1]->src[1].value << run[j - 1]->shift) >
              ((uint64_t)run[j]->src[1].value << run[j]->shift); --j) {
            agx_instr *lower = run[j];
            list_del(&lower->link);
            list_addtail(&lower->link, &run[j - 1]->link);
            run[j] = run[j - 1];
            run[j - 1] = lower;
            progress = true;
         }
      }
   }

   return progress;
}

/* A structure read member by member from one M5 address issues one device
 * load per word, unless NIR vectorized them: their address arithmetic differs
 * until the address is lowered (base + u2u64(x + c) for each member), and
 * if-conversion may only bring them into one block afterwards. Once lowered,
 * such loads share their base, index and scale and differ in the byte offset
 * the hardware adds after scaling, so a vector load reads the same words at
 * the same addresses. Merge loads of consecutive words that no memory write,
 * atomic, barrier or discard separates into one, placed at the first.
 */
static bool
agx3_m5_word_load(const agx_instr *I)
{
   /* Immediate indices are left to agx3_order_fragment_device_loads */
   return I->op == AGX_OPCODE_DEVICE_LOAD && I->nr_dests == 1 &&
          I->nr_srcs == 2 && I->src[1].type == AGX_INDEX_NORMAL &&
          I->format == AGX_FORMAT_I32 && I->mask == 1 &&
          !I->memory_is_packed && I->dest[0].type == AGX_INDEX_NORMAL &&
          !I->dest[0].memory && I->dest[0].size == AGX_SIZE_32 &&
          agx_channels(I->dest[0]) == 1;
}

static bool
agx3_m5_same_word_address(const agx_instr *A, const agx_instr *B)
{
   for (unsigned s = 0; s < 2; ++s) {
      if (!agx_is_equiv(A->src[s], B->src[s]) ||
          A->src[s].size != B->src[s].size ||
          A->src[s].abs != B->src[s].abs || A->src[s].neg != B->src[s].neg)
         return false;
   }

   return A->shift == B->shift && A->coherent == B->coherent &&
          A->memory_index_is_bytes == B->memory_index_is_bytes &&
          A->scoreboard == B->scoreboard;
}

struct agx3_word {
   agx_instr *load;
   unsigned order;
};

static int
agx3_cmp_word_offset(const void *a_, const void *b_)
{
   const struct agx3_word *a = a_, *b = b_;
   const int16_t x = a->load->memory_byte_offset;
   const int16_t y = b->load->memory_byte_offset;
   return (x > y) - (x < y);
}

/* Merge the runs of consecutive words in a window of word loads, which no
 * memory write or barrier separates, given in program order.
 */
static bool
agx3_merge_word_window(agx_context *ctx, agx_instr **window,
                       const unsigned *position, unsigned n)
{
   bool progress = false;
   BITSET_DECLARE(done, 64) = {0};
   assert(n <= 64);

   for (unsigned i = 0; i < n; ++i) {
      if (BITSET_TEST(done, i))
         continue;

      struct agx3_word group[64];
      unsigned m = 0;
      for (unsigned j = i; j < n; ++j) {
         if (!BITSET_TEST(done, j) &&
             agx3_m5_same_word_address(window[i], window[j])) {
            group[m++] = (struct agx3_word){window[j], position[j]};
            BITSET_SET(done, j);
         }
      }

      qsort(group, m, sizeof(group[0]), agx3_cmp_word_offset);

      for (unsigned a = 0; a < m;) {
         const int base = group[a].load->memory_byte_offset;
         unsigned count = 1, earliest = a;
         unsigned lo = group[a].order, hi = group[a].order;
         while (a + count < m && count < 4 &&
                group[a + count].load->memory_byte_offset ==
                   base + 4 * (int)count) {
            /* Loading words far apart together would keep them all live */
            const unsigned order = group[a + count].order;
            if (MAX2(hi, order) - MIN2(lo, order) > 2 * (count + 1))
               break;

            lo = MIN2(lo, order);
            hi = MAX2(hi, order);
            if (order < group[earliest].order)
               earliest = a + count;
            count++;
         }

         /* Registers hold vectors in power-of-two slots: three words would
          * occupy four.
          */
         if (count == 3) {
            count = 2;
            if (earliest == a + 2) {
               earliest = group[a].order < group[a + 1].order ? a : a + 1;
            }
         }

         if (count > 1) {
            agx_instr *I = group[earliest].load;
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_index vec = agx_vec_temp(ctx, AGX_SIZE_32, count);
            agx_instr *load = agx_device_load_to(
               &b, vec, I->src[0], I->src[1], I->format, BITFIELD_MASK(count),
               I->shift, I->coherent);
            load->memory_index_is_bytes = I->memory_index_is_bytes;
            load->memory_byte_offset = base;
            load->memory_can_reorder = true;
            load->scoreboard = I->scoreboard;

            agx_instr *split = agx_split(&b, count, vec);
            for (unsigned c = 0; c < count; ++c) {
               agx_instr *word = group[a + c].load;
               split->dest[c] = word->dest[0];
               load->memory_can_reorder &= word->memory_can_reorder;
               agx_remove_instruction(word);
            }

            progress = true;
         }

         a += count;
      }
   }

   return progress;
}

static bool
agx3_merge_m5_word_loads(agx_context *ctx)
{
   /* Vertex and compute shaders often run just below an occupancy step,
    * which the vectors' wider register alignment can cross.
    */
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       ctx->stage != MESA_SHADER_FRAGMENT)
      return false;

   /* A word a phi reads is better loaded alone, straight into the register
    * the phi web wants.
    */
   BITSET_WORD *phi_source = BITSET_CALLOC(ctx->alloc);
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, phi) {
         agx_foreach_ssa_src(phi, s)
            BITSET_SET(phi_source, phi->src[s].value);
      }
   }

   bool progress = false;
   agx_foreach_block(ctx, block) {
      /* Word loads since the last memory write or barrier, in order */
      agx_instr *window[64];
      unsigned position[64];
      unsigned n = 0, ip = 0;

      agx_foreach_instr_in_block_safe(block, I) {
         ip++;
         const enum agx_schedule_class cls =
            agx_opcodes_info[I->op].schedule_class;
         const bool end = cls == AGX_SCHEDULE_CLASS_STORE ||
                          cls == AGX_SCHEDULE_CLASS_ATOMIC ||
                          cls == AGX_SCHEDULE_CLASS_BARRIER ||
                          cls == AGX_SCHEDULE_CLASS_COVERAGE ||
                          cls == AGX_SCHEDULE_CLASS_INVALID ||
                          instr_after_logical_end(I);

         const bool word = agx3_m5_word_load(I) &&
                           !BITSET_TEST(phi_source, I->dest[0].value);

         if (!end && word && n < ARRAY_SIZE(window)) {
            position[n] = ip;
            window[n++] = I;
            continue;
         }

         if (!end && !word)
            continue;

         progress |= agx3_merge_word_window(ctx, window, position, n);
         n = 0;

         if (!end && word) {
            position[n] = ip;
            window[n++] = I;
         }
      }

      progress |= agx3_merge_word_window(ctx, window, position, n);
   }

   free(phi_source);
   return progress;
}

/* Spill reloads that repeat a device load (agx_spill) can leave such loads
 * adjacent again after register allocation, possibly in descending order.
 * Restore the ascending order once registers are final, where two adjacent
 * loads may only trade places if neither reads or writes what the other
 * writes. Format and mask need not match: native code ascends regardless.
 */
static bool
agx3_ordered_load_candidate(const agx_instr *I)
{
   return I->op == AGX_OPCODE_DEVICE_LOAD && !I->coherent &&
          I->nr_dests == 1 && I->nr_srcs == 2 &&
          I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].abs &&
          !I->src[1].neg && !I->src[1].memory;
}

static int64_t
agx3_ordered_load_address(const agx_instr *I)
{
   unsigned width_log2 = 0;
   if (I->format == AGX_FORMAT_I16 || I->format == AGX_FORMAT_F16)
      width_log2 = 1;
   else if (I->format == AGX_FORMAT_I32)
      width_log2 = 2;

   const unsigned shift =
      I->shift + (I->memory_index_is_bytes ? 0 : width_log2);
   return ((int64_t)I->src[1].value << shift) + I->memory_byte_offset;
}

static bool
agx3_registers_overlap(agx_index a, agx_index b)
{
   if (a.memory || b.memory ||
       (a.type != AGX_INDEX_REGISTER && a.type != AGX_INDEX_NORMAL) ||
       (b.type != AGX_INDEX_REGISTER && b.type != AGX_INDEX_NORMAL))
      return false;

   return a.value < b.value + agx_index_size_16(b) &&
          b.value < a.value + agx_index_size_16(a);
}

static bool
agx3_loads_independent(const agx_instr *A, const agx_instr *B)
{
   if (agx3_registers_overlap(A->dest[0], B->dest[0]))
      return false;

   for (unsigned s = 0; s < 2; ++s) {
      if (agx3_registers_overlap(A->dest[0], B->src[s]) ||
          agx3_registers_overlap(B->dest[0], A->src[s]))
         return false;
   }

   return true;
}

static void
agx3_order_fragment_device_loads_after_ra(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       ctx->stage != MESA_SHADER_FRAGMENT)
      return;

   agx_foreach_block(ctx, block) {
      bool progress;
      do {
         progress = false;
         agx_foreach_instr_in_block_safe(block, I) {
            if (I->link.next == &block->instructions)
               break;

            agx_instr *next = list_entry(I->link.next, agx_instr, link);
            if (!agx3_ordered_load_candidate(I) ||
                !agx3_ordered_load_candidate(next) ||
                !agx_is_equiv(I->src[0], next->src[0]) ||
                agx3_ordered_load_address(I) <=
                   agx3_ordered_load_address(next) ||
                !agx3_loads_independent(I, next))
               continue;

            /* The later load is now the last use of a shared base */
            const bool kill = next->src[0].kill;
            next->src[0].kill = I->src[0].kill;
            I->src[0].kill = kill;

            list_del(&next->link);
            list_addtail(&next->link, &I->link);
            progress = true;
         }
      } while (progress);
   }
}

/* Keep scalar UVS values live until a native-shaped terminal writeout.
 * Run after pressure scheduling but before RA: moving physical-register stores
 * would lose values when later instructions reuse their sources. */
bool
agx3_sink_vertex_outputs(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       ctx->stage != MESA_SHADER_VERTEX || ctx->is_preamble ||
       ctx->key->secondary)
      return false;

   /* Outputs already belonging to the final block can be bundled locally
    * even when earlier blocks contain instance transforms. NIR may also
    * leave the stores in the immediately preceding block and put only STOP
    * in a synthetic final block. Treat that single-predecessor,
    * single-successor fallthrough as one terminal region: moving uses over
    * that edge cannot change their execution mask. */
   agx_block *previous = NULL, *stop_block = NULL;
   agx_foreach_block(ctx, block) {
      previous = stop_block;
      stop_block = block;
   }
   if (stop_block == NULL)
      return false;

   agx_block *output_block = stop_block;
   bool final_has_store = false;
   agx_foreach_instr_in_block(stop_block, I)
      final_has_store |= I->op == AGX_OPCODE_ST_VARY;

   if (!final_has_store && previous &&
       agx_num_successors(previous) == 1 &&
       previous->successors[0] == stop_block &&
       agx_num_predecessors(stop_block) == 1 &&
       *util_dynarray_element(&stop_block->predecessors, agx_block *, 0) ==
          previous &&
       previous->divergent == stop_block->divergent) {
      output_block = previous;
   }

   agx_foreach_block(ctx, block) {
      if (block == output_block)
         break;
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_ST_VARY || I->op == AGX_OPCODE_STOP)
            return false;
      }
   }

   agx_instr *stores[AGX3_MAX_VERTEX_OUTPUTS] = {0}, *stop = NULL;
   if (output_block != stop_block) {
      agx_foreach_instr_in_block(stop_block, I) {
         if (I->op != AGX_OPCODE_STOP || stop)
            return false;
         stop = I;
      }
   }

   bool seen_stop = false;
   agx_foreach_instr_in_block(output_block, I) {
      if (seen_stop)
         return false;
      switch (I->op) {
      case AGX_OPCODE_STOP:
         if (output_block != stop_block || stop)
            return false;
         stop = I;
         seen_stop = true;
         break;
      case AGX_OPCODE_ST_VARY: {
         if (I->nr_dests || I->nr_srcs != 2 ||
             I->src[0].type != AGX_INDEX_IMMEDIATE || I->src[0].value >= ARRAY_SIZE(stores) ||
             I->src[1].type != AGX_INDEX_NORMAL || I->src[1].memory ||
             I->src[1].size != AGX_SIZE_32 || agx_channels(I->src[1]) != 1 ||
             stores[I->src[0].value])
            return false;
         stores[I->src[0].value] = I;
         break;
      }
      case AGX_OPCODE_ATOMIC:
         /* UVS is invocation-private output storage, disjoint from device
          * memory. Delay only its stores, retaining the relative order of
          * every atomic and memory access. SSA output values remain live
          * until the terminal bundle because this runs before RA. Barriers,
          * other side effects and control flow still refuse below. */
         break;
      default:
         /* Pure SSA producers may move relative to vertex writeout. Their
          * actual encodings remain validated by the normal compiler path;
          * other side effects and control flow require an ordering proof. */
         if (!agx_opcodes_info[I->op].can_eliminate)
            return false;
         break;
      }
   }
   if (!stop)
      return false;
   /* Constant position components need no UVS store. A layer or point-size
    * export can therefore follow holes in the position vector. Sort the live
    * SSA exports just as for dense outputs, without manufacturing values for
    * those holes. The packer's sparse protocol retains their actual indices.
    */
   bool user_output = false;
   for (unsigned i = 4; i < ARRAY_SIZE(stores); ++i)
      user_output |= stores[i] != NULL;
   if (!user_output)
      return false;
   for (unsigned i = 0; i < ARRAY_SIZE(stores); ++i) {
      if (!stores[i])
         continue;
      list_del(&stores[i]->link);
      list_addtail(&stores[i]->link, &stop->link);
   }
   return true;
}

static bool
agx3_vertex_alu_operand(agx_index x, bool allow_uniform)
{
   if (x.size != AGX_SIZE_32 || agx_channels(x) != 1 || x.abs || x.neg ||
       x.memory)
      return false;
   /* The staged FFMA names its uniform with an 8-bit half index. */
   return x.type == AGX_INDEX_NORMAL ||
          (allow_uniform && x.type == AGX_INDEX_UNIFORM && x.value < 256);
}

/* Write a vertex output's FMUL or FFMA straight into the staging file and
 * store it from there, as native vertex programs do (agx3_stage_vertex_alu:
 * a staged FMUL, a uniform*register+register FFMA and an all-register FFMA),
 * instead of computing it into a GPR and copying that into the staging file.
 * Each fusion removes one COPY. The ALU moves down to the store, where its
 * register sources are still available: SSA sources dominate their use.
 *
 * Only a producer whose sole use is the store is fused, and only operands the
 * native forms take: scalar 32-bit registers without modifiers, and for FFMA
 * a uniform first multiplicand (the multiplicands commute). Run after
 * agx3_sink_vertex_outputs, which recognizes plain stores only. */
static bool
agx3_fuse_vertex_output_alu(agx_context *ctx)
{
   /* ST_VARY_FMA/MUL_AGX3 are the G17P vertex-output forms; native M5
    * terrain vertex mains use the same three (agx3_stage_vertex_alu). */
   if ((ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B &&
        ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C) ||
       ctx->stage != MESA_SHADER_VERTEX || ctx->is_preamble)
      return false;

   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
   uint32_t *uses = calloc(ctx->alloc, sizeof(*uses));
   if (!defs || !uses) {
      free(defs);
      free(uses);
      return false;
   }

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
      agx_foreach_ssa_src(I, s)
         uses[I->src[s].value]++;
   }

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_ST_VARY || I->nr_srcs != 2)
         continue;

      const agx_index value = I->src[1];
      if (value.type != AGX_INDEX_NORMAL || value.size != AGX_SIZE_32 ||
          agx_channels(value) != 1 || value.abs || value.neg ||
          value.memory || uses[value.value] != 1)
         continue;

      agx_instr *P = defs[value.value];
      if (!P || (P->op != AGX_OPCODE_FMUL && P->op != AGX_OPCODE_FFMA) ||
          P->nr_dests != 1 || P->saturate ||
          P->dest[0].size != AGX_SIZE_32 || agx_channels(P->dest[0]) != 1)
         continue;

      const bool fma = P->op == AGX_OPCODE_FFMA;
      agx_index a = P->src[0], b = P->src[1];
      if (fma && b.type == AGX_INDEX_UNIFORM && a.type != AGX_INDEX_UNIFORM) {
         agx_index t = a;
         a = b;
         b = t;
      }
      if (!agx3_vertex_alu_operand(a, fma) ||
          !agx3_vertex_alu_operand(b, false) ||
          (fma && !agx3_vertex_alu_operand(P->src[2], false)))
         continue;

      agx_builder bld = agx_init_builder(ctx, agx_before_instr(I));
      if (fma)
         agx_st_vary_fma_agx3(&bld, I->src[0], a, b, P->src[2]);
      else
         agx_st_vary_mul_agx3(&bld, I->src[0], a, b);

      agx_remove_instruction(I);
      agx_remove_instruction(P);
      progress = true;
   }

   free(defs);
   free(uses);
   return progress;
}

/* Reordering tile loads must not move them above their sample-mask definitions.
 * Uniform-source legalization can leave separate MOV/SPLIT chains next to each
 * load, even when every load uses the same logical mask. Move only the pure
 * dependencies within this group, in their original SSA order. In particular,
 * an ordered coverage read must stay at its original execution point.
 */
static bool
agx3_hoist_tile_load_sources(agx_context *ctx, agx_instr **loads, unsigned count)
{
   BITSET_WORD *needed = calloc(BITSET_WORDS(ctx->alloc), sizeof(*needed));
   agx_instr **hoist = calloc(ctx->alloc, sizeof(*hoist));
   unsigned nr_hoist = 0;
   bool valid = true;

   for (unsigned i = 0; i < count; ++i) {
      agx_foreach_ssa_src(loads[i], s)
         BITSET_SET(needed, loads[i]->src[s].value);
   }

   /* The input array is still in program order. Build the complete movement
    * plan before changing the list, so an immovable dependency leaves it alone.
    */
   struct list_head *end = loads[0]->link.prev;
   for (struct list_head *node = &loads[count - 1]->link; node != end;
        node = node->prev) {
      agx_instr *I = list_entry(node, agx_instr, link);
      bool required = false;
      agx_foreach_ssa_dest(I, d)
         required |= BITSET_TEST(needed, I->dest[d].value);

      if (!required)
         continue;

      const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
      if (!info->can_eliminate || !info->can_reorder ||
          info->schedule_class != AGX_SCHEDULE_CLASS_NONE) {
         valid = false;
         break;
      }

      agx_foreach_src(I, s) {
         if (I->src[s].type == AGX_INDEX_REGISTER)
            valid = false;
      }
      if (!valid)
         break;

      agx_foreach_ssa_src(I, s)
         BITSET_SET(needed, I->src[s].value);
      hoist[nr_hoist++] = I;
   }

   if (valid) {
      while (nr_hoist) {
         agx_instr *I = hoist[--nr_hoist];
         list_del(&I->link);
         list_addtail(&I->link, &loads[0]->link);
      }
   }

   free(hoist);
   free(needed);
   return valid;
}

/* Apple's G17P multi-RT fragments finish colour partitions in descending
 * order and make partition zero the final release.  This holds for both the
 * opaque two-RGBA8 witness (offsets 4, 0) and the mixed RGBA16F+R8 witnesses
 * (offsets 8, 0), including programmable blending.  Sink the stores while
 * they are still in SSA so every payload remains live through the native
 * ordering and RA cannot reuse an earlier target's physical register.
 *
 * This is partition-driven, not format- or workload-driven.  It is restricted
 * to straight-line fragment epilogs and internal background programs, and it
 * refuses any non-colour side effect other than a leading depth export.  The
 * packer independently proves the partition graph before selecting the
 * corresponding pixel classes.
 */
#define AGX3_ORDER_REFUSE()                                                    \
   do {                                                                      \
      if (agx_compiler_debug & AGX_DBG_REFUSE)                                 \
         fprintf(stderr, "G17P MRT order refused at %s:%u\n", __FILE__,         \
                 __LINE__);                                                  \
      return false;                                                          \
   } while (0)

static bool
agx3_order_g18_msaa_tile_ops(agx_context *ctx)
{
   /* Native454/513/567: raw32, packed BGRA8 and RGBA16Float tile accesses
    * use their actual tuple size. Retain complete sample groups; only
    * move stores within their block, after every group definition. */
   BITSET_DECLARE(offsets, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   unsigned stores = 0, loads = 0;
   unsigned tile_stride = 0;
   agx_index coverage = agx_null();
   bool common_coverage = true;
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_ST_TILE && I->op != AGX_OPCODE_LD_TILE)
         continue;
      const unsigned stride = I->format == AGX_FORMAT_I32
                                 ? (I->mask == 1 ? 4 : I->mask == 15 ? 16 : 0)
                                 : I->mask != 15 ? 0
                                 : I->format == AGX_FORMAT_U8NORM ? 4
                                 : I->format == AGX_FORMAT_F16 ? 8 : 0;
      if (!stride ||
          (tile_stride && tile_stride != stride) ||
          I->explicit_coords || (I->pixel_offset % stride) ||
          I->pixel_offset >= AGX_G17P_MAX_BYTES_PER_SAMPLE)
         AGX3_ORDER_REFUSE();
      tile_stride = stride;
      if (I->op == AGX_OPCODE_ST_TILE) {
         BITSET_SET(offsets, I->pixel_offset);
         stores++;
         if (agx_is_null(coverage))
            coverage = I->src[1];
         else
            common_coverage &= agx_is_equiv(coverage, I->src[1]);
      } else {
         loads++;
      }
   }
   const unsigned partitions = __bitset_count(offsets, ARRAY_SIZE(offsets));
   /* Pixel-shaded pure stores can broadcast one colour per target through
    * a common coverage mask. Keep tile reads on the per-sample path. */
   const bool broadcast = partitions >= 2 && stores == partitions && !loads &&
                          common_coverage && list_length(&ctx->blocks) == 1;
   if (!partitions || partitions > AGX_MAX_RTS ||
       (!broadcast && stores != partitions * ctx->key->fs.nr_samples))
      AGX3_ORDER_REFUSE();
   if (broadcast) {
      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_ST_TILE || I->op == AGX_OPCODE_STOP)
            continue;
         if ((I->op == AGX_OPCODE_WAIT_PIX ||
              I->op == AGX_OPCODE_SIGNAL_PIX) && I->writeout == 12)
            continue;
         if (!agx_opcodes_info[I->op].can_eliminate)
            AGX3_ORDER_REFUSE();
      }
   }
   for (unsigned i = 0; i < partitions; ++i) {
      if (!BITSET_TEST(offsets, tile_stride * i))
         AGX3_ORDER_REFUSE();
   }
   agx_foreach_block(ctx, block) {
      agx_instr *group[AGX_MAX_RTS * 4];
      unsigned count = 0;
      agx_foreach_instr_in_block(block, I) {
         if (I->op != AGX_OPCODE_ST_TILE)
            continue;
         if (count == ARRAY_SIZE(group))
            AGX3_ORDER_REFUSE();
         group[count++] = I;
      }
      if (count % partitions)
         AGX3_ORDER_REFUSE();
      for (unsigned base = 0; base < count; base += partitions) {
         struct list_head *anchor = group[base + partitions - 1]->link.next;
         for (unsigned i = base; i < base + partitions; ++i) {
            for (unsigned j = i + 1; j < base + partitions; ++j) {
               if (group[j]->pixel_offset > group[i]->pixel_offset)
                  SWAP(group[i], group[j]);
            }
         }
         for (unsigned i = base; i < base + partitions; ++i) {
            if (group[i]->pixel_offset !=
                tile_stride * (partitions - 1 - (i - base)))
               AGX3_ORDER_REFUSE();
            list_del(&group[i]->link);
            list_addtail(&group[i]->link, anchor);
         }
      }
   }
   /* The packer holds the combined resident classes across the complete
    * sample group and releases after its final store. Generic per-store
    * releases would close that lifetime early. Preserve depth signals. */
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block_safe(block, I) {
         if (I->op == AGX_OPCODE_SIGNAL_PIX && I->writeout == 12)
            agx_remove_instruction(I);
      }
   }
   ctx->agx3_mrt_tile_ops_ordered = true;
   return true;
}

static bool
agx3_order_g17p_mrt_tile_ops(agx_context *ctx)
{
   /* G18G: this optional pass cannot affect a non-fragment program or a
    * preamble. Returning false means no transformation, not a refused shader.
    * Avoid reporting these ordinary calls as failed MRT admission. */
   if (agx_is_g18g(&ctx->key->dev) &&
       (ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble))
      return false;

   if (agx_is_g18g(&ctx->key->dev) && ctx->stage == MESA_SHADER_FRAGMENT &&
       ctx->key->secondary && !ctx->key->fs.ignore_tib_dependencies &&
       (ctx->key->fs.nr_samples == 2 || ctx->key->fs.nr_samples == 4))
      return agx3_order_g18_msaa_tile_ops(ctx);
   /* G18's native two-target fragment uses the same descending stores and
    * per-target class pairs. Keep its qualification to one-sample shaders. */
   const bool g18 = agx_is_g18g(&ctx->key->dev) && ctx->key->fs.nr_samples <= 1;
   if (agx_is_g18g(&ctx->key->dev) && !g18)
      return false;

   agx_block *tail = g18 ? NULL : agx_reconverged_mrt_tail(ctx);
   if (ctx->key->g17p_emrt && ctx->key->secondary &&
       agx_isa_is_agx3(ctx->key->dev.encoding) &&
       ctx->stage == MESA_SHADER_FRAGMENT && ctx->key->fs.nr_samples >= 1 &&
       list_length(&ctx->blocks) > 1 && !tail) {
      /* Keep each sample within its block. Sorting stores before that
       * group's final anchor preserves SSA dominance and never moves a
       * value across conditional image accesses in another block. */
      const unsigned partitions = __bitset_count(
         ctx->key->g17p_tile_partitions,
         BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE));
      if (partitions >= 2) {
         agx_foreach_block(ctx, block) {
            agx_instr *group[AGX_MAX_RTS * 8];
            unsigned count = 0;
            agx_foreach_instr_in_block(block, I) {
               if (I->op != AGX_OPCODE_ST_TILE)
                  continue;
               if (count == ARRAY_SIZE(group))
                  return false;
               group[count++] = I;
            }
            if (count % partitions)
               return false;
            for (unsigned base = 0; base < count; base += partitions) {
               struct list_head *anchor = group[base + partitions - 1]->link.next;
               for (unsigned i = base; i < base + partitions; ++i) {
                  for (unsigned j = i + 1; j < base + partitions; ++j) {
                     if (group[j]->pixel_offset > group[i]->pixel_offset)
                        SWAP(group[i], group[j]);
                  }
               }
               for (unsigned i = base; i < base + partitions; ++i) {
                  list_del(&group[i]->link);
                  list_addtail(&group[i]->link, anchor);
               }
            }
         }
      }

      /* The packer independently checks counts, formats and every group. */
      unsigned previous = UINT_MAX, stores = 0;
      agx_foreach_instr_global(ctx, I) {
         if (I->op != AGX_OPCODE_ST_TILE)
            continue;
         if (I->pixel_offset >= previous)
            return false;
         previous = I->pixel_offset ? I->pixel_offset : UINT_MAX;
         stores++;
      }
      if (stores < 2 || previous != UINT_MAX)
         return false;
      ctx->agx3_mrt_tile_ops_ordered = true;
      return true;
   }

   if ((ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B &&
        !(ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          ctx->key->fs.nr_samples <= 8)) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       (!g18 && !ctx->key->secondary && !ctx->key->fs.ignore_tib_dependencies &&
        (ctx->key->no_stop ||
         !(agx_get_fs_structure_flags() & AGX_FS_STRUCTURE_INLINE_BLEND))) ||
       (list_length(&ctx->blocks) != 1 && !tail))
      AGX3_ORDER_REFUSE();

   /* A multisample epilog stores every target through the coverage word
    * register (agx_nir_lower_g17p_tile_samples). */
   const bool multisample_stores =
      ctx->key->fs.nr_samples > 1 && !ctx->key->fs.ignore_tib_dependencies;
   agx_instr *loads[AGX_MAX_RTS * 8] = {0};
   agx_instr *stores[AGX_MAX_RTS * 8] = {0};
   agx_instr *stop = NULL;
   agx_instr *first_wait = NULL;
   unsigned load_count = 0, store_count = 0;
   /* Apple's store-only MRT fragments that export depth complete the class-1
    * depth transaction before the first colour transaction, which follows
    * unchanged. Admit exactly that leading wait, depth write and signal. */
   unsigned depth_ops = 0, sample_masks = 0;
   agx_foreach_block(ctx, block) {
      if (tail && block != tail)
         continue;
      agx_foreach_instr_in_block(block, I) {
         const bool colour_started = first_wait || load_count || store_count;
         switch (I->op) {
         case AGX_OPCODE_PRELOAD:
         case AGX_OPCODE_COLLECT:
         case AGX_OPCODE_SPLIT:
            break;
         case AGX_OPCODE_WAIT_PIX:
            if (I->writeout == 1) {
               if (depth_ops != 0 || colour_started)
                  return false;
               depth_ops = 1;
               break;
            }
            /* Native sample loops keep the depth transaction across each
             * sample's colour accesses. The generated eMRT epilog has the
             * same lifetime even when its whole layout fits in the tile. */
            if (depth_ops == 1 ||
                (depth_ops == 2 &&
                 !((ctx->key->g17p_emrt || agx3_g17p_interleaved_zs(ctx)) &&
                   multisample_stores)))
               return false;
            if (!first_wait)
               first_wait = I;
            break;
         case AGX_OPCODE_ZS_EMIT_AGX3:
            /* A sample-shaded epilog writes depth/stencil once per unrolled
             * sample inside the same class-1 transaction. */
            if (depth_ops != 1 && depth_ops != 2 &&
                !(ctx->key->g17p_emrt && multisample_stores && depth_ops == 3))
               return false;
            depth_ops = 2;
            break;
         case AGX_OPCODE_SIGNAL_PIX:
            if (depth_ops != 2 || I->writeout != 1)
               return false;
            depth_ops = 3;
            break;
         case AGX_OPCODE_FLTNORM_AGX3:
            break;
         case AGX_OPCODE_SAMPLE_MASK_AGX3:
            /* A force-early shader defers its sample mask to the epilog,
             * which kills the masked samples before anything else runs:
             * agx_nir_prolog_epilog.c emits the demote from the imported
             * write-samples word, and the packer places the instruction ahead
             * of the first WAIT_PIX. That is outside the colour transaction,
             * so it does not disturb the class protocol. Admit it only there;
             * between tile ops it would kill samples mid-transaction. */
            if (colour_started || depth_ops || sample_masks++)
               return false;
            break;
         case AGX_OPCODE_ST_TILE:
            if (store_count >= ARRAY_SIZE(stores) || I->nr_dests != 0 ||
                I->nr_srcs != 3 || I->explicit_coords ||
                I->src[0].type != AGX_INDEX_NORMAL || I->src[0].memory ||
                !(multisample_stores
                     ? I->src[1].type == AGX_INDEX_NORMAL &&
                          I->src[1].size == AGX_SIZE_16
                     : I->src[1].type == AGX_INDEX_IMMEDIATE &&
                          I->src[1].value == 0xff) ||
                !agx_is_null(I->src[2]))
               return false;
            if (!multisample_stores) {
               for (unsigned i = 0; i < store_count; ++i) {
                  if (stores[i]->pixel_offset == I->pixel_offset)
                     return false;
               }
            }
            stores[store_count++] = I;
            break;
         case AGX_OPCODE_LD_TILE:
            if (load_count >= ARRAY_SIZE(loads) || I->nr_dests != 1 ||
                I->nr_srcs != 2 || I->explicit_coords ||
                !(multisample_stores
                     ? I->src[0].type == AGX_INDEX_NORMAL &&
                          I->src[0].size == AGX_SIZE_16
                     : I->src[0].type == AGX_INDEX_IMMEDIATE &&
                          I->src[0].value == 0xff) ||
                !agx_is_null(I->src[1]))
               return false;
            if (!multisample_stores) {
               for (unsigned i = 0; i < load_count; ++i) {
                  if (loads[i]->pixel_offset == I->pixel_offset)
                     return false;
               }
            }
            loads[load_count++] = I;
            break;
         case AGX_OPCODE_STOP:
            if (stop)
               return false;
            stop = I;
            break;
         default:
            if (!agx_opcodes_info[I->op].can_eliminate)
               return false;
            break;
         }
      }
   }

   /* An epilog writing only some of the framebuffer's targets keeps each
    * store's partition. A lone store to partition zero is the ordinary
    * single-target protocol. */
   const BITSET_WORD *partitions = ctx->key->g17p_tile_partitions;
   const bool keyed = !__bitset_is_empty(
      partitions, BITSET_WORDS(AGX_G17P_MAX_BYTES_PER_SAMPLE));
   BITSET_DECLARE(written, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   bool unkeyed_store = false;
   for (unsigned i = 0; i < store_count; ++i) {
      const unsigned offset = stores[i]->pixel_offset;
      if (offset >= AGX_G17P_MAX_BYTES_PER_SAMPLE)
         return false;
      BITSET_SET(written, offset);
      unkeyed_store |= !BITSET_TEST(partitions, offset);
   }
   const bool subset =
      keyed && memcmp(written, partitions, sizeof(written)) != 0;
   const bool only_partition0 =
      __bitset_count(written, ARRAY_SIZE(written)) == 1 &&
      BITSET_TEST(written, 0);
   /* An unkeyed GL store still names its logical attachment. Keep the
    * qualified Vulkan sample-shaded load/depth protocol alongside it. */
   const bool lone_sparse_store =
      !keyed && store_count == 1 && stores[0]->colour_target != 0;
   /* A keyed subset may blend each stored partition while omitted partitions
    * remain untouched. Apple's four-RGBA32Float G17P witness with RT0 masked
    * and RT1..3 blended uses exactly that load/store shape. */
   if (!stop || (keyed && unkeyed_store) ||
       (!lone_sparse_store &&
        store_count < (subset && !only_partition0 ? 1 : 2)) ||
       (depth_ops && depth_ops != 3))
      return false;

   /* Native G18 RGBA8 fragments with two through eight attachments, including
    * sparse writes and blending, load and store in descending target order.
    * Each class's read acquisition remains held through its final store.
    * Native596 R32Uint/RGBA8 witnesses also finish a leading class-1 depth
    * transaction before the first store-only colour transaction. The scan
    * above requires its wait/export/signal to finish before colour starts.
    * Keep depth plus framebuffer fetch separate: native loads can precede
    * depth, which needs a different ordering proof. Sample masks and
    * backgrounds likewise remain separate protocols. */
   /* Backgrounds use the separately validated transaction-free protocol;
    * this optional fragment ordering pass must not transform them. */
   if (g18 && ctx->key->fs.ignore_tib_dependencies)
      return false;
   if (g18 && (!store_count || store_count > AGX_MAX_RTS ||
                sample_masks || (depth_ops && load_count)))
       AGX3_ORDER_REFUSE();

   /* Native raw32 BG clear/reload keeps ascending stores and has no
    * ordinary colour transactions, including two/three RGBA32 targets. Other
    * multi-target BG programs have no transactions either and store in no
    * required partition order.
    *
    * The wide-MRT class protocol is a FRAGMENT witness, and extending it to
    * clears is not supported by anything: a background compiled with the
    * transaction no longer matches agx3_g17p_rt0_bg_program_generate()'s
    * loader descriptor, and KHR-GL30.draw_buffers.draw_buffers_1 aborts the
    * process at "background Ioto program" on real hardware. */
   if (ctx->key->fs.ignore_tib_dependencies) {
      bool rgba32 = !load_count;
      for (unsigned i = 0; i < store_count; ++i)
         rgba32 &= stores[i]->format == AGX_FORMAT_I32 &&
                   stores[i]->mask == 15 && stores[i]->pixel_offset == 16 * i;
      bool rgba8 = !load_count && store_count >= 3;
      for (unsigned i = 0; i < store_count; ++i)
         rgba8 &= stores[i]->format == AGX_FORMAT_U8NORM &&
                  stores[i]->mask == 15 && stores[i]->pixel_offset == 4 * i;
      if (ctx->key->g17p_mrt_bg || store_count >= 4 || rgba32 || rgba8)
         return false;
   }

   if (load_count && !first_wait)
      return false;

   /* Constant sample masks can remove any subset of the unrolled samples.
    * Count the complete surviving target sets, not the framebuffer samples.
    * Their masks and relative sample order remain unchanged.
    */
   const unsigned partition_count =
      __bitset_count(written, ARRAY_SIZE(written));
   if (!partition_count || store_count % partition_count)
      return false;
   const unsigned sample_groups = store_count / partition_count;
   if (sample_groups > (multisample_stores ? ctx->key->fs.nr_samples : 1))
      return false;

   /* Native acquires and loads higher partitions first.  Move only the tile
    * loads themselves; their SSA consumers stay in place and RA extends the
    * values' physical lifetimes safely. */
   if (!multisample_stores) {
      for (unsigned i = 0; i < load_count; ++i) {
         for (unsigned j = i + 1; j < load_count; ++j) {
            if (loads[j]->pixel_offset > loads[i]->pixel_offset) {
               agx_instr *tmp = loads[i];
               loads[i] = loads[j];
               loads[j] = tmp;
            }
         }
      }

      agx_instr *load_anchor = first_wait;
      for (unsigned i = 0; i < load_count; ++i) {
         list_del(&loads[i]->link);
         list_add(&loads[i]->link, &load_anchor->link);
         load_anchor = loads[i];
      }
   } else if (load_count) {
      const unsigned samples = sample_groups;
      if (load_count % samples)
         return false;

      const unsigned group_size = load_count / samples;
      for (unsigned group = 0; group < samples; ++group) {
         agx_instr **group_loads = &loads[group * group_size];
         bool ordered = true;
         for (unsigned i = 1; i < group_size; ++i)
            ordered &= group_loads[i - 1]->pixel_offset >
                       group_loads[i]->pixel_offset;

         /* A tile load may have its own ordered coverage read immediately
          * before it. Keep an already ordered group in place instead of
          * hoisting the later loads above their mask definitions. */
         if (ordered) {
            for (unsigned i = 0; group > 0 && i < group_size; ++i) {
               if (group_loads[i]->pixel_offset != loads[i]->pixel_offset)
                  return false;
            }
            continue;
         }
         if (!agx3_hoist_tile_load_sources(ctx, group_loads, group_size))
            return false;
         struct list_head *anchor = group_loads[0]->link.prev;

         for (unsigned i = 0; i < group_size; ++i) {
            for (unsigned j = i + 1; j < group_size; ++j) {
               if (group_loads[j]->pixel_offset >
                   group_loads[i]->pixel_offset) {
                  agx_instr *tmp = group_loads[i];
                  group_loads[i] = group_loads[j];
                  group_loads[j] = tmp;
               }
            }

            if (group > 0 &&
                group_loads[i]->pixel_offset != loads[i]->pixel_offset)
               return false;
         }

         for (unsigned i = 0; i < group_size; ++i) {
            list_del(&group_loads[i]->link);
            list_add(&group_loads[i]->link, anchor);
            anchor = &group_loads[i]->link;
         }
      }
   }

   /* Sort physical tile partitions by byte offset, highest first. A G17P
    * sample-shaded epilog is unrolled into one complete target set per
    * sample. Keep those sample sets separate while giving each the native
    * descending partition order. */
   const bool unrolled_samples = multisample_stores && sample_groups > 1;
   const unsigned group_count = sample_groups;
   const unsigned group_size = store_count / group_count;
   struct list_head *group_anchor[8] = {0};
   for (unsigned group = 0; group < group_count; ++group) {
      agx_instr **group_stores = &stores[group * group_size];
      if (unrolled_samples)
         group_anchor[group] = group_stores[group_size - 1]->link.next;

      for (unsigned i = 0; i < group_size; ++i) {
         for (unsigned j = i + 1; j < group_size; ++j) {
            if (group_stores[j]->pixel_offset >
                group_stores[i]->pixel_offset) {
               agx_instr *tmp = group_stores[i];
               group_stores[i] = group_stores[j];
               group_stores[j] = tmp;
            }
         }

         if (unrolled_samples && group > 0 &&
             group_stores[i]->pixel_offset != stores[i]->pixel_offset)
            return false;
      }
   }
   /* Native1130 (run 1139): a G18 mixed-format fragment stores nothing for a
    * target whose write mask is empty, so its lowest store may sit above
    * byte zero (five of the 39 fragments). */
   if (stores[store_count - 1]->pixel_offset != 0 && !subset &&
       !(g18 && ctx->key->g18_native_mixed))
      return false;

   if (unrolled_samples) {
      /* Keep each sample's payload calculation and stores together. Moving
       * every sample's stores to STOP would make all exported colours live at
       * once and exceed the grounded tile-store register bank at 4x. */
      for (unsigned group = 0; group < group_count; ++group) {
         for (unsigned i = 0; i < group_size; ++i) {
            agx_instr *store = stores[group * group_size + i];
            list_del(&store->link);
            list_addtail(&store->link, group_anchor[group]);
         }
      }
   } else {
      for (unsigned i = 0; i < store_count; ++i) {
         list_del(&stores[i]->link);
         list_addtail(&stores[i]->link, &stop->link);
      }
   }
   ctx->agx3_mrt_tile_ops_ordered = true;
   return true;
}

/* The tile load whose value a store writes back unchanged, or NULL. The value
 * is the load's result or the collect of its split channels in order. */
static const agx_instr *
agx3_unchanged_tile_load(agx_instr **defs, const agx_instr *store)
{
   const agx_index value = store->src[0];
   if (value.type != AGX_INDEX_NORMAL || !defs[value.value])
      return NULL;

   const agx_instr *def = defs[value.value];
   if (def->op == AGX_OPCODE_COLLECT) {
      const agx_instr *split = NULL;
      agx_foreach_src(def, s) {
         const agx_index c = def->src[s];
         const agx_instr *c_def =
            c.type == AGX_INDEX_NORMAL ? defs[c.value] : NULL;
         if (!c_def || c_def->op != AGX_OPCODE_SPLIT ||
             (split && c_def != split) || c_def->nr_dests != def->nr_srcs ||
             !agx_is_equiv(c_def->dest[s], c))
            return NULL;
         split = c_def;
      }
      if (!split || split->src[0].type != AGX_INDEX_NORMAL)
         return NULL;
      def = defs[split->src[0].value];
   }

   return def && def->op == AGX_OPCODE_LD_TILE ? def : NULL;
}

/* A colour write disabled on a format without masked stores reaches the
 * epilog as a load of the target stored back unchanged. Apple's G17P
 * fragments store nothing to a channel their write mask excludes
 * (blend/native/tilemask.py wm_*), so drop such stores; the loads then die.
 */
static bool
agx3_drop_unchanged_tile_stores(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       list_length(&ctx->blocks) != 1)
      return false;

   agx_instr **defs = calloc(ctx->alloc, sizeof(*defs));
   if (!defs)
      return false;

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d)
         defs[I->dest[d].value] = I;
   }

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_ST_TILE || I->explicit_coords ||
          I->src[1].type != AGX_INDEX_IMMEDIATE)
         continue;

      const agx_instr *load = agx3_unchanged_tile_load(defs, I);
      if (!load || load->pixel_offset != I->pixel_offset ||
          load->format != I->format || load->mask != I->mask ||
          load->explicit_coords || !agx_is_equiv(load->src[0], I->src[1]))
         continue;

      /* No other store to the target may come between the two. */
      bool clobbered = false;
      for (const agx_instr *J = list_entry(load->link.next, agx_instr, link);
           J != I; J = list_entry(J->link.next, agx_instr, link)) {
         clobbered |= J->op == AGX_OPCODE_ST_TILE &&
                      J->pixel_offset == I->pixel_offset;
      }
      if (clobbered)
         continue;

      agx_remove_instruction(I);
      progress = true;
   }

   free(defs);
   return progress;
}

/* A read of a target whose value turns out to be dead still leaves the
 * class-8 read wait emitted before it, and the store that follows waits only
 * for the remaining class 4. Apple's store-only G17P fragments, including
 * those writing only some channels, take class 12 at once, and those writing
 * no channel take no colour class at all (blend/native/tilemask.py wm_*,
 * blend/native/wm/none.py). With no tile read left, fold each read wait into
 * the next write wait of its block, or drop the colour waits when nothing is
 * stored either.
 */
bool
agx3_merge_unread_pixel_waits(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       ctx->key->fs.ignore_tib_dependencies || ctx->any_interlock)
      return false;

   bool stores = false;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_LD_TILE)
         return false;
      stores |= I->op == AGX_OPCODE_ST_TILE;
   }

   bool progress = false;
   agx_foreach_block(ctx, block) {
      agx_instr *read_wait = NULL;
      agx_foreach_instr_in_block_safe(block, I) {
         if (I->op != AGX_OPCODE_WAIT_PIX)
            continue;

         if (!stores) {
            if (I->writeout == 0x4 || I->writeout == 0x8 ||
                I->writeout == 0xc) {
               agx_remove_instruction(I);
               progress = true;
            }
         } else if (I->writeout == 0x8 && !read_wait) {
            read_wait = I;
         } else if (I->writeout == 0x4 && read_wait) {
            I->writeout = 0xc;
            agx_remove_instruction(read_wait);
            read_wait = NULL;
            progress = true;
         }
      }
   }

   return progress;
}

/* Tile stores read their payloads from a bank of GPR halves
 * (agx3_tile_store_bank). Payloads that must be live together beyond the
 * grounded bank have no register assignment: refuse the shader rather than
 * let register allocation find none.
 */
static bool
agx3_tile_store_payloads_fit(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble)
      return true;

   const unsigned bank = agx3_tile_store_bank(ctx);
   unsigned *uses = calloc(ctx->alloc, sizeof(*uses));
   unsigned *widths = calloc(ctx->alloc, sizeof(*widths));
   if (!uses || !widths) {
      free(uses);
      free(widths);
      return false;
   }

   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_ST_TILE && I->src[0].type == AGX_INDEX_NORMAL) {
         uses[I->src[0].value]++;
         widths[I->src[0].value] = agx_index_size_16(I->src[0]);
      }
   }

   bool fits = true;
   unsigned halves = 0;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d) {
         unsigned ssa = I->dest[d].value;
         if (uses[ssa])
            halves += widths[ssa];
      }

      if (I->op != AGX_OPCODE_ST_TILE)
         continue;

      /* A CSE'd payload used by several samples occupies one register
       * vector, kept live until its final store, not one vector per store. */
      fits &= halves <= bank;

      if (I->src[0].type == AGX_INDEX_NORMAL &&
          --uses[I->src[0].value] == 0)
         halves -= widths[I->src[0].value];
   }

   free(uses);
   free(widths);
   return fits;
}

/* Scalar epilog inputs retain their fixed export ABI, and unlike a collect a
 * direct PRELOAD cannot be assigned into a tile store's register bank by RA:
 * the output of colour target n arrives at GPR half 8 + 8n. The grounded
 * stores read halves 0..31 in store-only fragments of three and four targets
 * (and multi-target BG programs), halves 0..63 with more targets, and halves
 * 0..15 otherwise, so a store of the second target alone found its input out
 * of bank and was refused.
 * Materialize only out-of-bank scalar inputs, immediately before the first
 * store, and let RA choose a legal free register. The bank mirrors RA's
 * tile_gpr_limit.
 */
static bool
agx3_bank_scalar_tile_store_inputs(agx_context *ctx)
{
   if (!agx_isa_is_agx3(ctx->key->dev.encoding) ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble)
      return false;

   agx_instr *first_store = NULL;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_ST_TILE) {
         first_store = first_store ? first_store : I;
      }
   }
   if (!first_store)
      return false;

   const unsigned bank = agx3_tile_store_bank(ctx);

   bool progress = false;
   agx_builder b = agx_init_builder(ctx, agx_before_instr(first_store));
   agx_foreach_instr_global(ctx, store) {
      /* Test the opcode before reading src[0]: an instruction with no
       * sources (e.g. stop) has no src[0]. */
      if (store->op != AGX_OPCODE_ST_TILE)
         continue;
      agx_index source = store->src[0];
      if (source.type != AGX_INDEX_NORMAL || agx_channels(source) != 1)
         continue;

      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_PRELOAD && I->nr_dests == 1 &&
             agx_is_equiv(I->dest[0], source)) {
            if (I->src[0].type == AGX_INDEX_REGISTER &&
                I->src[0].value >= bank) {
               agx_index copy = agx_temp(ctx, source.size);
               agx_mov_to(&b, copy, source);
               store->src[0] = copy;
               progress = true;
            }
            break;
         }
      }
   }
   return progress;
}

static bool
agx3_implicit_coord_stage_producer(const agx_instr *I)
{
   if (I == NULL || I->nr_dests != 1 || I->dest[0].type != AGX_INDEX_NORMAL ||
       I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
       I->saturate)
      return false;

   /* These are the two native terminal-coordinate forms needed by ordinary
    * smooth sampling and compositor UV+offset sampling. Other arithmetic
    * stays on the established GPR plus COPY path until its staging destination
    * form is grounded independently. */
   return I->op == AGX_OPCODE_PERSP_MUL_AGX3 ||
          (I->op == AGX_OPCODE_FADD && I->nr_srcs == 2 &&
           I->src[0].type == AGX_INDEX_NORMAL &&
           I->src[1].type == AGX_INDEX_NORMAL &&
           I->src[0].size == AGX_SIZE_32 &&
           I->src[1].size == AGX_SIZE_32 &&
           agx_channels(I->src[0]) == 1 &&
           agx_channels(I->src[1]) == 1 && !I->src[0].abs &&
           !I->src[0].neg && !I->src[1].abs && !I->src[1].neg);
}

/* G17G: a texture sample that leaves its coordinate vector live (one varying
 * sampled from two textures) reads wrong coordinates: both samples return the
 * corner texel. Samples that consume their coordinates are correct. Give each
 * sample whose coordinates have another use a private copy it consumes. */
static bool
agx3c_private_tex_coords(agx_context *ctx)
{
   if (ctx->key->dev.chip != AGX_CHIP_G17G)
      return false;

   unsigned *uses = rzalloc_array(ctx, unsigned, ctx->alloc);
   if (uses == NULL)
      return false;
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s)
         uses[I->src[s].value]++;
   }

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_TEXTURE_SAMPLE ||
          I->src[0].type != AGX_INDEX_NORMAL || uses[I->src[0].value] < 2)
         continue;
      const unsigned n = agx_channels(I->src[0]);
      if (n < 1 || n > 4)
         continue;
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index comps[4];
      agx_emit_split(&b, comps, I->src[0], n);
      agx_index copy = agx_vec_temp(ctx, I->src[0].size, n);
      agx_emit_collect_to(&b, copy, n, comps);
      uses[I->src[0].value]--;
      I->src[0] = copy;
      progress = true;
   }
   ralloc_free(uses);
   return progress;
}

/* Mark the last two scalar coordinate operations of a straight-line implicit
 * fragment sample for direct staging destinations. Apple does not calculate
 * the UVs in GPRs and copy them afterwards: the final perspective MUL or ADD
 * itself publishes staging[0]/[2]. That distinction preserves helper-lane
 * provenance at a misaligned hardware-scissor edge, which is required for the
 * implicit derivative quad and therefore the correct mip level.
 *
 * Run after scheduling so the start marker names the actual first producer,
 * but before RA while exact SSA def/use identity is still available. */
static bool
agx3_mark_implicit_tex_coord_staging(agx_context *ctx)
{
   /* AGX3B only: the staging-destination PERSP_MUL form is G17P's. */
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       ctx->key->fs.ignore_tib_dependencies || list_length(&ctx->blocks) != 1)
      return false;

   agx_instr **defs = rzalloc_array(ctx, agx_instr *, ctx->alloc);
   unsigned *uses = rzalloc_array(ctx, unsigned, ctx->alloc);
   if (defs == NULL || uses == NULL)
      return false;

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s) {
         assert(I->src[s].value < ctx->alloc);
         uses[I->src[s].value]++;
      }
      agx_foreach_ssa_dest(I, d) {
         assert(I->dest[d].value < ctx->alloc);
         defs[I->dest[d].value] = I;
      }
   }

   bool progress = false;
   agx_foreach_instr_global(ctx, sample) {
      const bool plain_implicit =
         sample->op == AGX_OPCODE_TEXTURE_SAMPLE && sample->nr_srcs == 6 &&
         sample->lod_mode == AGX_LOD_MODE_AUTO_LOD &&
         sample->dim == AGX_DIM_2D && !sample->shadow && !sample->offset &&
         !sample->sparse &&
         sample->gather == AGX_GATHER_NONE;
      if (!plain_implicit || sample->src[0].type != AGX_INDEX_NORMAL ||
          sample->src[0].size != AGX_SIZE_32 ||
          agx_channels(sample->src[0]) != 2 ||
          uses[sample->src[0].value] != 1)
         continue;

      agx_instr *collect = defs[sample->src[0].value];
      if (collect == NULL || collect->op != AGX_OPCODE_COLLECT ||
          collect->nr_srcs != 2 || collect->nr_dests != 1)
         continue;

      agx_instr *producer[2] = {NULL, NULL};
      bool valid = true;
      for (unsigned c = 0; c < 2; ++c) {
         const agx_index scalar = collect->src[c];
         if (scalar.type != AGX_INDEX_NORMAL || scalar.size != AGX_SIZE_32 ||
             agx_channels(scalar) != 1 || uses[scalar.value] != 1) {
            valid = false;
            break;
         }

         producer[c] = defs[scalar.value];
         if (!agx3_implicit_coord_stage_producer(producer[c]) ||
             producer[c]->agx3_tex_coord_stage) {
            valid = false;
            break;
         }
      }

      if (!valid || producer[0] == producer[1])
         continue;

      /* Apple's G17P implicit fragment emits every coefficient interpolation
       * before it opens the coordinate-staging transaction, then publishes X
       * and Y consecutively. Keep an unrelated interpolation out of that
       * bounded staging transaction, exactly as in the native witness. Both
       * terminal producers are pure, their SSA results have the single
       * COLLECT use proven above, and sinking them to the COLLECT therefore
       * preserves data dependencies.
       */
      for (unsigned c = 0; c < 2; ++c) {
         list_del(&producer[c]->link);
         list_addtail(&producer[c]->link, &collect->link);
      }

      for (unsigned c = 0; c < 2; ++c) {
         producer[c]->agx3_tex_coord_stage = true;
         producer[c]->agx3_tex_coord_component = c;
      }
      producer[0]->agx3_tex_coord_start = true;
      sample->agx3_tex_coords_staged = true;
      progress = true;
      /* One synthetic staging token is modeled by the slot planner. A later
       * sample retains the existing immediate COPY path; broad overlapping
       * staging groups need distinct native lifetime evidence. */
      break;
   }

   return progress;
}

/* Match Asahi's output-temporary model before selecting physical tile
 * writes. Repeated and conditional whole-pixel stores are shader outputs,
 * whose final value is written once at the reconverged exit. Tile reads and
 * explicit/per-sample stores keep their original ordering. */
static bool
agx_nir_fragment_output_temporaries(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   struct {
      bool seen;
      unsigned base, format, mask;
      nir_variable *var;
   } rt[8] = {0};
   nir_variable *exports[256] = {0};
   bool has_tile_store = false, has_export = false, needed = false;
   unsigned tile_writes[8] = {0}, export_writes[256] = {0};
   unsigned export_bits[256] = {0};
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx)
               return false;
            if (intr->intrinsic == nir_intrinsic_export_agx) {
               const unsigned bits = intr->src[0].ssa->bit_size;
               if ((bits != 32 && bits != 16) ||
                   intr->src[0].ssa->num_components != 1 ||
                   nir_intrinsic_base(intr) >= ARRAY_SIZE(exports))
                  return false;
               unsigned index = nir_intrinsic_base(intr);
               if (export_bits[index] && export_bits[index] != bits)
                  return false;
               export_bits[index] = bits;
               needed |= export_writes[index]++ != 0 ||
                  block != nir_impl_last_block(impl);
               if (!exec_node_is_tail_sentinel(instr->node.next)) {
                  nir_instr *next = nir_instr_next(instr);
                  needed |= next->type != nir_instr_type_intrinsic ||
                     nir_instr_as_intrinsic(next)->intrinsic != nir_intrinsic_export_agx;
               }
               has_export = true;
            }
            if (intr->intrinsic != nir_intrinsic_store_local_pixel_agx)
               continue;
            const unsigned target = nir_intrinsic_target(intr);
            if (target >= ARRAY_SIZE(rt) || nir_intrinsic_explicit_coord(intr) ||
                intr->src[0].ssa->bit_size != 32 ||
                !nir_src_is_const(intr->src[1]) ||
                nir_src_as_uint(intr->src[1]) != 0xff)
               return false;
            const unsigned base = nir_intrinsic_base(intr);
            const unsigned format = nir_intrinsic_format(intr);
            if (rt[target].seen &&
                (rt[target].base != base || rt[target].format != format))
               return false;
            needed |= tile_writes[target]++ != 0 ||
               block != nir_impl_last_block(impl);
            rt[target].seen = true;
            rt[target].base = base;
            rt[target].format = format;
            rt[target].mask |= nir_intrinsic_write_mask(intr);
            has_tile_store = true;
         }
      }
   }
   if ((!has_tile_store && !has_export) || !needed)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   for (unsigned i = 0; i < ARRAY_SIZE(rt); ++i) {
      if (rt[i].seen)
         rt[i].var = nir_local_variable_create(impl, glsl_vec4_type(), NULL);
   }
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_local_pixel_agx) {
            unsigned target = nir_intrinsic_target(intr);
            b.cursor = nir_before_instr(instr);
            nir_store_var(&b, rt[target].var,
                          nir_pad_vector(&b, intr->src[0].ssa, 4),
                          nir_intrinsic_write_mask(intr));
            nir_instr_remove(instr);
         } else if (intr->intrinsic == nir_intrinsic_export_agx) {
            unsigned index = nir_intrinsic_base(intr);
            if (!exports[index])
               exports[index] = nir_local_variable_create(
                  impl, export_bits[index] == 16 ? glsl_float16_t_type()
                                                 : glsl_float_type(),
                  NULL);
            b.cursor = nir_before_instr(instr);
            nir_store_var(&b, exports[index], intr->src[0].ssa, 1);
            nir_instr_remove(instr);
         }
      }
   }
   b.cursor = nir_after_impl(impl);
   for (unsigned i = 0; i < ARRAY_SIZE(rt); ++i) {
      if (rt[i].seen)
         nir_store_local_pixel_agx(
            &b, nir_load_var(&b, rt[i].var), nir_imm_intN_t(&b, 0xff, 16),
            nir_undef(&b, 2, 16), .base = rt[i].base, .target = i,
            .format = rt[i].format, .write_mask = rt[i].mask);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(exports); ++i) {
      if (exports[i])
         nir_export_agx(&b, nir_load_var(&b, exports[i]), .base = i);
   }
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_phis_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_opt_dce);
   return true;
}

/* Fold 8-to-32 unsigned conversion into the selected result type. The
 * four-byte transaction is still determined by the format and lane count.
 * agx_emit_load() then extracts the lanes from the byte-packed word that a
 * four-lane W8 load returns on AGX3B.
 */
static bool
fold_w8_word_unpack(nir_builder *b, nir_intrinsic_instr *load,
                    UNUSED void *data)
{
   if (b->impl->function->is_preamble ||
       (load->intrinsic != nir_intrinsic_load_agx &&
        load->intrinsic != nir_intrinsic_load_constant_agx) ||
       load->def.bit_size != 8 || load->def.num_components != 4 ||
       nir_intrinsic_format(load) != PIPE_FORMAT_R8_UINT ||
       (nir_intrinsic_base(load) & (AGX_NIR_MEMORY_PACKED_COUNT_MASK |
                                   AGX_NIR_MEMORY_UNPACK_W8_WORD)) ||
       nir_is_coherent(load))
      return false;

   bool used = false;
   nir_foreach_use_including_if(use, &load->def) {
      if (nir_src_is_if(use))
         return false;
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->type != nir_instr_type_alu)
         return false;
      const nir_alu_instr *alu = nir_instr_as_alu(parent);
      if (alu->op != nir_op_u2u32 || alu->def.bit_size != 32 ||
          alu->def.num_components != 1 || alu->src[0].swizzle[0] >= 4)
         return false;
      used = true;
   }
   if (!used)
      return false;

   load->def.bit_size = 32;
   nir_intrinsic_set_base(load, nir_intrinsic_base(load) |
                               AGX_NIR_MEMORY_UNPACK_W8_WORD);
   nir_foreach_use(use, &load->def)
      nir_instr_as_alu(nir_src_use_instr(use))->op = nir_op_mov;

   return true;
}

/* This pass runs after all NIR speculation and movement. The actual global
 * load is introduced only within its valid arm, and cannot be speculated or
 * reordered. The frontend predicate retains its original overflow semantics. */
static bool
m5_lower_predicated_load(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global_predicated_agx)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   enum gl_access_qualifier access = nir_intrinsic_access(intr) &
      ~(ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   nir_push_if(b, intr->src[2].ssa);
   nir_def *value = nir_load_global(b, intr->num_components, intr->def.bit_size,
      nir_iadd(b, intr->src[0].ssa, nir_u2u64(b, intr->src[1].ssa)),
      .access = access, .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr));
   nir_push_else(b, NULL);
   nir_def *zero = nir_imm_zero(b, intr->num_components, intr->def.bit_size);
   nir_pop_if(b, NULL);
   nir_def_replace(&intr->def, nir_if_phi(b, value, zero));
   return true;
}

static bool
agx_compile_function_nir_attempt(nir_shader *nir, nir_function_impl *impl,
                         struct agx_shader_key *key,
                         struct util_dynarray *binary,
                         struct agx_shader_info *out, unsigned *offset_out,
                         unsigned ra_budget_cut, bool disable_remat, bool *retry_without_remat,
                         bool agx3_reg_float_consts)
{
   const unsigned initial_binary_size = binary->size;
   const struct agx_shader_info initial_out = *out;
   if (agx_isa_is_agx3(key->dev.encoding) &&
       !impl->function->is_preamble && !ra_budget_cut &&
       !agx3_reg_float_consts)
      NIR_PASS(_, nir, agx_nir_fragment_output_temporaries);

   if (agx_isa_is_agx3(key->dev.encoding) &&
       nir->info.stage == MESA_SHADER_VERTEX && !impl->function->is_preamble &&
       exec_list_length(&impl->body) > 1 && !ra_budget_cut &&
       !agx3_reg_float_consts)
      NIR_PASS(_, nir, nir_move_output_stores_to_end);

   nir_index_blocks(impl);
   nir_index_ssa_defs(impl);

   agx_context *ctx = rzalloc(NULL, agx_context);
   ctx->nir = nir;
   ctx->ra_budget_cut = ra_budget_cut;
   ctx->is_preamble = impl->function->is_preamble;
   ctx->agx3_reg_float_consts = agx3_reg_float_consts;
   if (ctx->is_preamble)
      out->agx3_preamble_static_spill_version = 0;
   ctx->out = out;
   ctx->key = key;
   ctx->stage = nir->info.stage;
   ctx->allocated_vec = _mesa_hash_table_u64_create(ctx);
   ctx->indexed_nir_blocks = rzalloc_array(ctx, agx_block *, impl->num_blocks);
   list_inithead(&ctx->blocks);

   if (ctx->stage == MESA_SHADER_FRAGMENT && !ctx->is_preamble) {
      ctx->any_cf = key->fs.inside_sample_loop;
      ctx->force_early_tests_pending = nir->info.fs.early_fragment_tests;

      /* A G17G epilog runs inside the pixel transaction its main opened:
       * agx_nir_fs_epilog marks a main that holds the read class. */
      if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          nir->info.internal && nir->info.fs.pixel_interlock_ordered)
         ctx->already_pixel_waited = 0x8;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            ctx->zs_writes_left +=
               instr->type == nir_instr_type_intrinsic &&
               nir_instr_as_intrinsic(instr)->intrinsic ==
                  nir_intrinsic_store_zs_agx &&
               !(nir_intrinsic_base(nir_instr_as_intrinsic(instr)) &
                 AGX_ZS_COVERED_SAMPLE);
         }
      }
   }

   ctx->alloc = impl->ssa_alloc;
   /* AGX3 uniform control-flow hints need divergence metadata for this final
    * IR. AGX2 keeps upstream's conservative hints. */
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
      nir_divergence_analysis(nir);
   emit_cf_list(ctx, &impl->body);
   if (ctx->unsupported_instruction) {
      agx_set_refusal(ctx, "unsupported instruction", __LINE__);
      binary->size = initial_binary_size;
      ralloc_free(ctx);
      return false;
   }
   agx_emit_phis_deferred(ctx);

   /* Index blocks now that we're done emitting so the order is consistent. Do
    * this before agx_remove_unreachable_blocks so we match NIR indexing. This
    * makes for easier debugging.
    */
   agx_foreach_block(ctx, block) {
      block->index = ctx->num_blocks++;
   }

   agx_remove_unreachable_blocks(ctx);

   /* Only allocate scratch if it's statically used, regardless of if the NIR
    * info claims otherwise.
    */
   if (ctx->any_scratch) {
      assert(!ctx->is_preamble && "preambles don't use scratch");
      ctx->scratch_size_B = align(nir->scratch_size, 16);
   }

   /* Stop the main shader or preamble shader after the exit block. For real
    * functions, we would return here.
    */
   if (!ctx->key->no_stop || ctx->is_preamble) {
      agx_block *last_block = list_last_entry(&ctx->blocks, agx_block, link);
      agx_builder _b = agx_init_builder(ctx, agx_after_block(last_block));
      agx_stop(&_b);
   }

   agx_validate(ctx, "IR translation");

   if (likely(!(agx_compiler_debug & AGX_DBG_NOOPT))) {
      /* Eliminate dead instructions before CSE to avoid silly scheduling */
      AGX_PASS(ctx, agx_dce, false);

      /* CSE before eliminating dead destinations so that subdivision is
       * optimized properly.
       */
      AGX_PASS(ctx, agx_opt_cse);

      /* After DCE, use counts are right so we can run the optimizer. */
      AGX_PASS(ctx, agx_optimizer_backward);
      AGX_PASS(ctx, agx_optimizer_forward);
      AGX_PASS(ctx, agx_opt_compact_constants);

      /* After inlining constants, promote what's left */
      if (key->promote_constants && !key->secondary &&
          !(agx_compiler_debug & AGX_DBG_NOPROMOTE)) {
         AGX_PASS(ctx, agx_opt_promote_constants);
      }
   }

   agx_lower_agx3_half_converts(ctx);
   agx_lower_agx3_fma_sources(ctx);
   agx_lower_agx3_mixed_width_unary(ctx);
   agx_lower_agx3_shifted_iadd_dest(ctx);
   agx_lower_m5_alu_operands(ctx);

   /* Before materialization, which lowers the SIGNEXT this introduces. */
   AGX_PASS(ctx, agx_lower_agx3_widening_int);

   if (key->dev.encoding != AGX_ISA_ENCODING_AGX2)
      agx_materialize_agx3_alu_constants(ctx);

   /* M5 (AGX3C): an f16-destination FADD/FFMA whose sources are 32-bit
    * registers or immediates has no grounded encoding (the packer refused
    * `ffma d16, r32, #imm16, r32` in RayTracedGL1's IndirectInit raygen).
    * agx_optimizer's fold guard only sees immediates that exist at fold
    * time; a constant inlined afterwards by agx_opt_compact_constants slips
    * through.  Compute such an operation at its sources' width and narrow
    * the result with the explicit f32-to-f16 move the optimizer would have
    * kept, which rounds once exactly as the unfused sequence did. */
   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
      agx_foreach_instr_global_safe(ctx, I) {
         if ((I->op != AGX_OPCODE_FFMA && I->op != AGX_OPCODE_FADD) ||
             I->nr_dests != 1 || I->dest[0].size != AGX_SIZE_16 ||
             agx_channels(I->dest[0]) != 1)
            continue;

         bool wide = false, immediate = false;
         agx_foreach_src(I, s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE)
               immediate = true;
            else if (I->src[s].type != AGX_INDEX_NULL &&
                     I->src[s].size == AGX_SIZE_32)
               wide = true;
         }
         if (!(wide && immediate))
            continue;

         agx_index narrow = I->dest[0];
         agx_index result = agx_temp(ctx, AGX_SIZE_32);
         I->dest[0] = result;
         agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
         /* Saturation stays on the wide operation: clamping to [0, 1] and
          * then rounding to binary16 equals rounding then clamping, since
          * 0 and 1 are exact halves and rounding is monotonic. */
         agx_fmov_to(&b, narrow, result);
      }
   }

   /* Constant materialization can introduce mixed-width FADD operands. */
   agx_lower_agx3_fma_sources(ctx);
   agx3_fold_constant_fmovs(ctx);

   AGX_PASS(ctx, agx_lower_agx3_cf_constants);
   AGX_PASS(ctx, agx_lower_g17g_retry_float_operands);
   AGX_PASS(ctx, agx_lower_agx3_half_iadd_constants);
   AGX_PASS(ctx, agx_lower_agx3_wide_iadd);
   AGX_PASS(ctx, agx3_lower_large_fetch_indices);

   /* For correctness, lower uniform sources after copyprop (for correctness,
    * as copyprop creates uniform sources). To keep register pressure in
    * check, lower after CSE, since moves are cheaper than registers.
    */
   const bool native_cprog =
      key->g17p_native_uniforms || key->agx3_native_cprog;
   if (native_cprog && ctx->is_preamble &&
       agx_isa_is_agx3(key->dev.encoding)) {
      if (!agx3_lower_cprog_inputs(ctx)) {
         agx_set_refusal(ctx, "constant-program input lowering refused",
                         __LINE__);
         binary->size = initial_binary_size;
         ralloc_free(ctx);
         return false;
      }
      agx_validate(ctx, "agx3_lower_cprog_inputs");
      if (!(agx_compiler_debug & AGX_DBG_NOOPT))
         AGX_PASS(ctx, agx_opt_cse);
   }

   AGX_PASS(ctx, agx_lower_uniform_sources);

   /* Source legalization materializes the M5 float immediates no native form
    * reads in place; read them from constant uniforms where one does. */
   if (key->promote_constants && !key->secondary &&
       !(agx_compiler_debug & AGX_DBG_NOPROMOTE) &&
       likely(!(agx_compiler_debug & AGX_DBG_NOOPT)))
      AGX_PASS(ctx, agx_opt_promote_m5_float_constants);

   /* Legalization can introduce repeated half-to-float conversions and
    * uniform-source copies after the main CSE pass. Merge them before vertex
    * register allocation, retaining the legalized operand types. The existing
    * DMA path also needs this for descriptor words used by several operations.
    * Do not run copy propagation here: it could undo source legalization.
    * AGX3B main programs additionally use the pressure-aware CSE, and M5
    * main programs the bounded one described below. */
   const bool agx3_cse =
      (agx_isa_is_agx3(key->dev.encoding) &&
       ctx->stage == MESA_SHADER_VERTEX) ||
      (key->dev.encoding == AGX_ISA_ENCODING_AGX3B && !ctx->is_preamble) ||
      (key->g17p_uniform_dma && out->agx3_uniform_dma_count);
   const bool m5_bounded_cse =
      !agx3_cse && key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
      !ctx->is_preamble;
   if ((agx3_cse || m5_bounded_cse) &&
       !(agx_compiler_debug & AGX_DBG_NOOPT)) {
      /* AGX3B main programs use the pressure-aware CSE, which merges only
       * while the earlier result is still live. Uniform source lowering
       * copies a uniform into a GPR for each operation that cannot read it in
       * place, so a value used by a run of selects is copied once per select,
       * and each copy is read right after it is made. On G17P and in M5
       * main programs, also merge such copies where that keeps the register
       * pressure within the shader's existing peak. */
      if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B && !ctx->is_preamble &&
          ctx->stage != MESA_SHADER_VERTEX) {
         if (key->dev.chip == AGX_CHIP_G17P)
            AGX_PASS(ctx, agx_opt_cse_bounded);
         else
            AGX_PASS(ctx, agx_opt_cse_pressure);
      } else if (m5_bounded_cse) {
         AGX_PASS(ctx, agx_opt_cse_bounded);
      } else {
         AGX_PASS(ctx, agx_opt_cse);
      }
   }

   if (native_cprog && ctx->is_preamble &&
       agx_isa_is_agx3(key->dev.encoding)) {
      /* Selection requires a single live use of an arithmetic result.
       * Dead uses left by earlier lowering must not hide that opportunity. */
      AGX_PASS(ctx, agx_dce, true);
      AGX_PASS(ctx, agx3_lower_uniform_program);
   }

   AGX_PASS(ctx, agx3_coalesce_uniform_half_stores);
   if (native_cprog)
      AGX_PASS(ctx, agx3_lower_uniform_half_exports);
   AGX_PASS(ctx, agx3_drop_unchanged_tile_stores);

   AGX_PASS(ctx, agx_lower_agx3_immediates);

   AGX_PASS(ctx, agx3_merge_m5_word_loads);

   /* RA correctness depends on DCE */
   AGX_PASS(ctx, agx_dce, true);

   /* A preamble runs before the main shader and is not supplied with its
    * per-invocation GPR inputs. Check this after DCE so a dead preload does
    * not reject the shader, and before RA lowers PRELOAD away. This does
    * not qualify the AGX3 loader/constant-program entry ABI. G18G only. */
   if (ctx->is_preamble && agx_is_g18g(&ctx->key->dev)) {
      STATIC_ASSERT(AGX_NUM_UNIFORMS == AGX_IOTO_CPROG_UNIFORM_HALFS);
      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_PRELOAD) {
            if (agx_should_dump(nir, AGX_DBG_SHADERS))
               fprintf(stderr, "Preamble requires a main-shader entry GPR\n");
            ralloc_free(ctx);
            return false;
         }

      }
   }
   AGX_PASS(ctx, agx3_merge_unread_pixel_waits);

   AGX_PASS(ctx, agx_lower_agx3_device_atomic_sources);
   AGX_PASS(ctx, agx_lower_agx3_matrix_sources);

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      agx_print_shader(ctx, stdout);

   if (likely(!(agx_compiler_debug & AGX_DBG_NOSCHED))) {
      AGX_PASS(ctx, agx_pressure_schedule);
      if (agx_isa_is_agx3(ctx->key->dev.encoding))
         AGX_PASS(ctx, agx_ilp_schedule);
   }

   AGX_PASS(ctx, agx3_order_fragment_device_loads);

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      agx_print_shader(ctx, stdout);

   AGX_PASS(ctx, agx3_sink_vertex_outputs);
   AGX_PASS(ctx, agx3_fuse_vertex_output_alu);
   AGX_PASS(ctx, agx3_order_g17p_mrt_tile_ops);
   AGX_PASS(ctx, agx3_bank_scalar_tile_store_inputs);
   AGX_PASS(ctx, agx3_mark_implicit_tex_coord_staging);
   AGX_PASS(ctx, agx3c_private_tex_coords);

   /* Expose independent coordinate inputs before register allocation. Resource
    * values have no GPR destination; remove any now-unused coordinate collect.
    */
   if (agx3_lower_scalar_cube_coordinates(ctx)) {
      agx_validate(ctx, "scalar cube coordinates");
      AGX_PASS(ctx, agx_dce, true);
   }
   if (agx3_lower_scalar_texture_publications(ctx)) {
      agx_validate(ctx, "scalar texture publications");
      AGX_PASS(ctx, agx_dce, true);
   }
   if (agx3_split_texture_coordinates(ctx)) {
      agx_validate(ctx, "split texture coordinates");
      AGX_PASS(ctx, agx_dce, true);
   }


   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       agx_opt_collect_split(ctx)) {
      agx_validate(ctx, "collect/split folding");
      AGX_PASS(ctx, agx_dce, true);
   }

   AGX_PASS(ctx, agx3_fuse_fma_texture_publications);

   if (!agx3_tile_store_payloads_fit(ctx)) {
      agx_set_refusal(ctx, "tile-store payload bank overflow", __LINE__);
      ralloc_free(ctx);
      return false;
   }

   if (agx3_reserve_cube_scratch(ctx))
      agx_validate(ctx, "cube-array scratch");

   /* Interpolation tokens are checked on SSA: RA may move them. */
   if (agx_encoding_is_agx3(ctx->key->dev.encoding) &&
       ctx->stage == MESA_SHADER_FRAGMENT &&
       !agx3_interp_coords_from_itrcoords(ctx)) {
      agx_set_refusal(ctx, "interpolation token without its ITRCOORDS",
                      __LINE__);
      binary->size = initial_binary_size;
      ralloc_free(ctx);
      return false;
   }

   /* Record provenance before RA can create stack instructions. */
   const bool imports_only_before_ra = agx3_cprog_spill_provenance(ctx);
   if (disable_remat)
      agx_ra_without_remat(ctx);
   else
      agx_ra(ctx);

   /* A failed allocation is refused and retried below; its AIR is not a
    * program to validate (agx_validate exits on it). */
   if (!ctx->ra_failed)
      agx_validate(ctx, disable_remat ? "agx_ra_without_remat" : "agx_ra");

   /* The occupancy rematerialization attempt has already modified AIR.
    * Discard the entire context; changing its budget cannot undo spills. */
   if (retry_without_remat && ctx->remat_fallback_spilled) {
      *retry_without_remat = true;
      binary->size = initial_binary_size;
      ralloc_free(ctx);
      return false;
   }
   if (ctx->ra_failed) {
      /* Spilling only targets total demand; pinned values can still leave the
       * shuffle no room. Recompile with an ever smaller spill budget. */
      if (key->has_scratch && !key->is_helper && ra_budget_cut < 64) {
         ralloc_free(ctx);
         binary->size = initial_binary_size;
         /* Constant promotion and others accumulate into out. */
         *out = initial_out;
         return agx_compile_function_nir_attempt(
            nir, impl, key, binary, out, offset_out,
            ra_budget_cut ? ra_budget_cut * 2 : 16, disable_remat,
            retry_without_remat, agx3_reg_float_consts);
      }
      agx_set_refusal(ctx, "register allocation found no legal placement",
                      __LINE__);
      ralloc_free(ctx);
      return false;
   }

   /* Apple's G17P compiler publishes a 16-byte scratch footprint for compute,
    * vertex and fragment shaders that consume a device atomic's return.
    * The otherwise equivalent return-unused shaders publish zero, and the
    * returned form immediately reloads the old value from that thread-private
    * word. applegpu_g17p places the word directly above the spill frame: at
    * TPS half 0 without spills, and at half 130 after a kernel's spills end
    * at half 130. Keep this fixed ABI reservation distinct from
    * allocator-created spill size. */
   /* Software geometry and tessellation-control shaders keep their API
    * stage labels while running through CDM. Their returned atomics consume
    * the same thread-private word and require the same reservation.
    * AGX3B only: this is native G17P's ABI; on M5 the reservation makes the
    * precompiled helpers carry thread-private storage their launch refuses. */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_ATOMIC && I->nr_dests == 1 &&
             !agx_is_null(I->dest[0])) {
            const unsigned spill_frame_B = align(ctx->scratch_size_B, 4);

            ctx->agx3_atomic_return_tps_half = spill_frame_B / 2;
            out->agx3_spectpr_scratch_size =
               MAX2(out->agx3_spectpr_scratch_size, spill_frame_B + 16);
            break;
         }
      }
   }

   agx_lower_64bit_postra(ctx);
   agx_lower_agx3_stack_vectors(ctx);

   if (ctx->scratch_size_B > 0) {
      /* Apple always allocate 40 more bytes in the entrypoint and align to 4. */
      uint64_t stack_size = align(DIV_ROUND_UP(ctx->scratch_size_B, 4) + 10, 4);

      assert(stack_size < INT16_MAX);

      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
         agx_block *start_block = agx_start_block(ctx);
         agx_builder _b =
            agx_init_builder(ctx, agx_before_block(start_block));
         agx_stack_adjust(&_b, stack_size);

         /* If we're going to execute multiple times, make sure we clean up
          * after ourselves, else the hardware faults. */
         if (ctx->stage == MESA_SHADER_FRAGMENT && !ctx->is_preamble &&
             ctx->key->fs.inside_sample_loop) {
            _b = agx_init_builder(ctx, agx_after_block(agx_end_block(ctx)));
            agx_stack_adjust(&_b, -stack_size);
         }
      }

      /* AGX3 static spills are copy/rld accesses to the launch-allocated TPS
       * frame. The pinned G15/G16/G17 compute spill corpus emits no incStk;
       * that distinct instruction belongs to the ray-traversal stack. Keep
       * publishing the frame size below, but never inject an AGX2 stack
       * instruction into an AGX3 shader.
       *
       * G17P publishes that frame in bytes: applegpu_g17p's metadata for
       * kernels whose highest reloaded TPS half is 470 and 790 records 944
       * and 1584 bytes, the halves up to and including that 32-bit word.
       * SPECTPR's spill fields are those bytes in quads. */
      if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
         stack_size = align(ctx->scratch_size_B, 4);

      /* G18G native pressure launches publish the static TPS extent in bytes,
       * rounded to 16. COPY/RLD offsets address halves; SPECTPR divides this
       * byte count by four. The legacy word count and dynamic-stack header
       * do not describe this frame. M6 only: no M5 launch grounds it. */
      if (agx_is_g18g(&ctx->key->dev))
         stack_size = align(ctx->scratch_size_B, 16);

      if (ctx->is_preamble)
         out->preamble_scratch_size = stack_size;
      else
         out->scratch_size = stack_size;
   }

   if ((ctx->stage == MESA_SHADER_VERTEX ||
        ctx->stage == MESA_SHADER_TESS_EVAL) &&
       !impl->function->is_preamble && !ctx->key->secondary)
      agx_set_st_vary_final(ctx);

   /* AGX3 has no scoreboard `wait`; dependency fences ride on the terminal
    * `end`, and every real kernel ends with mask 0. Wait insertion is
    * therefore an AGX2-only pass. */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
      agx_insert_waits(ctx);
   agx_opt_empty_else(ctx);
   agx_opt_break_if(ctx);
   agx_opt_jmp_none(ctx);

   if (ctx->any_quad_divergent_shuffle)
      agx_lower_divergent_shuffle(ctx);

   agx_lower_pseudo(ctx);

   /* AGX3's thread-private storage moves one 16-bit or 32-bit component per
    * instruction, so a vector spill or scratch access has to become a run of
    * scalar ones. Do it before the last-use pass, so each component's hints
    * are computed against the instructions that actually survive. */
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
      agx3_split_vector_stack_access(ctx);

   /* Before the last-use pass too, which depends on instruction order */
   if (ctx->spilled_by_load)
      agx3_order_fragment_device_loads_after_ra(ctx);

   /* Set last-use bits after lowering pseudo for best results. */
   agx_opt_register_cache(ctx);

   /* Expose native publication events after physical GPR allocation. Keep
    * the established source hints on the publication that really reads them.
    * Like the other passes after agx_lower_pseudo(), this one is not
    * followed by agx_validate(): lowered loops no longer have SSA block form.
    */
   agx3_lower_texture_publications(ctx);

   if (ctx->is_preamble &&
       agx_isa_is_agx3(ctx->key->dev.encoding) &&
       !ctx->agx3_cprog_inputs_lowered &&
       !agx3_analyze_preamble_uniform_inputs(
          ctx, &out->agx3_preamble_uniform_inputs)) {
      agx_set_refusal(ctx, "constant-program uniform inputs not analysable "
                      "(no native CProg lowering ran)", __LINE__);
      binary->size = initial_binary_size;
      ralloc_free(ctx);
      return false;
   }

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      agx_print_shader(ctx, stdout);

   /* Upload constants before the binary instead after to reduce the chance they
    * get prefetched into the i-cache, when we want them only in the d-cache.
    * Also helps fill the padding space for small preambles.
    */
   if (ctx->out->rodata.size_16 && !impl->function->is_preamble) {
      ctx->out->rodata.offset = agx_pad_binary(binary, 4);
      unsigned size_16 = ctx->out->rodata.size_16;

      uint16_t *ro = util_dynarray_grow(binary, uint16_t, size_16);
      memcpy(ro, ctx->rodata, size_16 * 2);
   }

   unsigned offset = agx_pad_binary(binary, AGX_CODE_ALIGN);
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
      agx_pack_binary(ctx, binary);
   } else {
      /* AGX3 has its own native encoder. Device admission is checked separately
       * by agx_params_driver_supported(); G17P uses this path on hardware.
       * If the encoder refuses an instruction, discard the emitted prefix and
       * report compilation failure. */
      if (!agx3_pack_binary(ctx, binary)) {
         char what[768];
         snprintf(what, sizeof(what), "AGX3 binary packer refused %s",
                  agx3_last_pack_refusal()[0] ? agx3_last_pack_refusal()
                                              : "an instruction");
         agx_set_refusal(ctx, what, __LINE__);
         binary->size = initial_binary_size;
         ralloc_free(ctx);
         return false;
      }

      /* G18G: the G17G code just packed, re-encoded as agx3d. */
      const char *why;
      unsigned at;
      unsigned *entry = ctx->agx3_colour_link.entry_emitted
                           ? &ctx->agx3_colour_link.entry_offset : NULL;
      /* Run 1139: a G18 mixed-format fragment epilog's reads take the long
       * form when the packer emitted them through the native1130 protocol,
       * inside the 0x200 entry and final release. */
      const bool long_tile_reads = ctx->agx3_g18_mixed_protocol;
      if (ctx->key->dev.emit_agx3d &&
          !agx3_transcode_agx3d_tile_reads(binary, offset, entry,
                                          entry ? 1 : 0,
                                          ctx->stage == MESA_SHADER_FRAGMENT,
                                          long_tile_reads, &why, &at)) {
         if (agx_compiler_debug & AGX_DBG_REFUSE)
            fprintf(stderr, "G18G refused shader: %s at agx3c +0x%x\n", why,
                    at);
         binary->size = initial_binary_size;
         ralloc_free(ctx);
         return false;
      }
   }

   /* A relocatable vertex program lists its placeholder-index UVSWs just
    * after the main program. Every write of a placeholder index must be
    * listed, and each listed one must still hold its placeholder bytes, or
    * the linker could leave a varying at the wrong index. */
   if (ctx->key->g17p_uvs_reloc_size && !impl->function->is_preamble &&
       ctx->stage == MESA_SHADER_VERTEX) {
      const unsigned base = ctx->key->g17p_uvs_reloc_base;
      unsigned placeholders = 0;
      agx_foreach_instr_global(ctx, I) {
         if (agx_is_vertex_store(I) && I->src[0].type == AGX_INDEX_IMMEDIATE &&
             I->src[0].value >= base &&
             I->src[0].value - base < ctx->key->g17p_uvs_reloc_size)
            placeholders++;
      }

      uint8_t *relocs = ctx->agx3_uvs_relocs.data;
      for (unsigned i = 0; i < ctx->agx3_uvs_reloc_count; ++i) {
         uint32_t at;
         memcpy(&at, relocs + i * AGX3_UVS_RELOC_SIZE, sizeof(at));
         at -= offset;
         memcpy(relocs + i * AGX3_UVS_RELOC_SIZE, &at, sizeof(at));
      }

      if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
          placeholders != ctx->agx3_uvs_reloc_count ||
          !agx3_vary_relocate_uvsw(ctx->key->dev.encoding,
                                   (uint8_t *)binary->data + offset,
                                   binary->size - offset, relocs,
                                   ctx->agx3_uvs_reloc_count, NULL)) {
         agx_set_refusal(ctx, "relocatable UVS writes not all listed",
                         __LINE__);
         binary->size = initial_binary_size;
         ralloc_free(ctx);
         return false;
      }

      if (ctx->agx3_uvs_reloc_count) {
         out->agx3_uvs_relocs_offset = binary->size;
         out->agx3_uvs_reloc_count = ctx->agx3_uvs_reloc_count;
         util_dynarray_append_dynarray(binary, &ctx->agx3_uvs_relocs);
      }
   }

   /* Packing is complete: the certificate never describes an unencodable IR.
    * No machine-code or scheduling change is made by this analysis. */
   if (agx3_cprog_certify_static_spill(ctx, imports_only_before_ra))
      out->agx3_preamble_static_spill_version =
         ctx->stage == MESA_SHADER_FRAGMENT
            ? AGX3_CPROG_FS_STATIC_SPILL_VERSION
            : AGX3_CPROG_STATIC_SPILL_VERSION;

   unsigned nr_gprs = ctx->max_reg + 1;

   /* Preserve the legacy preamble scratch reservation on its existing
    * targets. AGX3B uses launch-allocated TPS with native COPY/RLD and no
    * AGX2 stack adjustment. Its input-storage prefix is not a reason to
    * reserve every GPR; report the allocation that actually reached packing. */
   if (impl->function->is_preamble) {
      const bool legacy_scratch = ctx->scratch_size_B &&
         !agx_isa_is_agx3(ctx->key->dev.encoding);
      out->nr_preamble_gprs = legacy_scratch ? 256 : nr_gprs;
      out->agx3_preamble_staging_halfs = ctx->agx3_staging_halfs;
      out->agx3_preamble_staging_known = ctx->agx3_staging_known;
   }
   else {
      out->nr_gprs = nr_gprs;
      out->agx3_staging_halfs = ctx->agx3_staging_halfs;
      out->agx3_staging_known = ctx->agx3_staging_known;
      out->agx3_fltnorm_start = ctx->agx3_fltnorm_start;
      out->agx3_colour_link = ctx->agx3_colour_link;
      out->agx3_pixel_held = ctx->already_pixel_waited;
   }

   if (impl->function->is_preamble) {
      uint32_t instrs = 0;
      agx_foreach_instr_global(ctx, I) {
         instrs++;
      }

      ctx->out->stats.preamble_inst = instrs;
   } else {
      agx_calc_stats(ctx, binary->size, &ctx->out->stats);
   }

   /*
    * Disassemble if the environment variable is set. Also disassemble to
    * /dev/null in debug builds as a smoke test - a legally encoded shader
    * should not hit any disassembler errors.
    */
   bool dump_shaders = agx_should_dump(nir, AGX_DBG_SHADERS);
   bool dump_binaries = dump_shaders || agx_should_dump(nir, AGX_DBG_BINARIES);
   if (dump_binaries && ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
      /* There is no AGX3 instruction printer here yet. Retain the exact
       * packed bytes alongside the IR for offline comparison and decoding.
       * Print each binary in one call so parallel compiler threads cannot
       * interleave the individual bytes. */
      const size_t size = binary->size - offset;
      char *hex = ralloc_array(ctx, char, size * 2 + 1);
      static const char digits[] = "0123456789abcdef";
      const uint8_t *code = (const uint8_t *)binary->data + offset;
      for (size_t i = 0; i < size; ++i) {
         hex[2 * i] = digits[code[i] >> 4];
         hex[2 * i + 1] = digits[code[i] & 0xf];
      }
      hex[2 * size] = '\0';
      fprintf(stderr,
              "AGX3 binary: stage=%u preamble=%u encoding=%u size=%zu hex=%s\n",
              ctx->stage, ctx->is_preamble, ctx->key->dev.encoding, size, hex);
   }
#ifndef NDEBUG
   bool selftest = !dump_shaders;
#else
   bool selftest = false;
#endif


   if ((dump_shaders || selftest) &&
       ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
      FILE *fp = selftest ? fopen("/dev/null", "w") : stdout;
      bool errors =
         agx2_disassemble(binary->data + offset, binary->size - offset, fp);

      if (errors) {
         /* In a self-test, disassemble again to get something the user can look
          * at to figure out what went wrong.
          */
         if (selftest) {
            agx2_disassemble(binary->data + offset, binary->size - offset,
                             stderr);
         }

         UNREACHABLE("Disassembly error hit.");
      }

      if (selftest) {
         fclose(fp);
      }
   }

   *offset_out = offset;
   ralloc_free(ctx);
   return true;
}

/* Retry eligible preambles without changing scratch admission or launch rules. */
static bool
agx_compile_function_nir(nir_shader *nir, nir_function_impl *impl,
                         struct agx_shader_key *key,
                         struct util_dynarray *binary,
                         struct agx_shader_info *out, unsigned *offset_out,
                         bool agx3_reg_float_consts)
{
   const bool eligible =
      impl->function->is_preamble && nir->info.stage == MESA_SHADER_FRAGMENT &&
      key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
      (key->g17p_native_uniforms || key->agx3_native_cprog) &&
      debug_get_bool_option("ASAHI_M5_CPROG_REMAT_RETRY", true);
   if (!eligible)
      return agx_compile_function_nir_attempt(nir, impl, key, binary, out,
                                             offset_out, 0, false, NULL,
                                             agx3_reg_float_consts);

   /* Preserve all compiler-owned state, including earlier main code/rodata.
    * The first attempt owns a deep NIR clone and a fresh AIR context. */
   nir_shader *attempt = nir_shader_clone(NULL, nir);
   if (!attempt)
      return agx_compile_function_nir_attempt(nir, impl, key, binary, out,
                                             offset_out, 0, false, NULL,
                                             agx3_reg_float_consts);

   nir_function_impl *attempt_impl = NULL;
   nir_foreach_function_with_impl(func, candidate, attempt) {
      if (func->is_preamble) {
         assert(attempt_impl == NULL);
         attempt_impl = candidate;
      }
   }
   assert(attempt_impl != NULL);
   const struct agx_shader_info saved_info = *out;
   const struct agx_shader_key saved_key = *key;
   const size_t saved_size = binary->size;
   bool retry = false;
   bool success = agx_compile_function_nir_attempt(
      attempt, attempt_impl, key, binary, out, offset_out, 0, false, &retry,
      agx3_reg_float_consts);
   ralloc_free(attempt);
   if (!retry)
      return success;

   assert(!success && binary->size == saved_size);
   *out = saved_info;
   *key = saved_key;
   binary->size = saved_size;
   /* Exactly one retry. Real full-budget spills retain the old refusal. */
   return agx_compile_function_nir_attempt(nir, impl, key, binary, out,
                                          offset_out, 0, true, NULL,
                                          agx3_reg_float_consts);
}

/*
 * The hardware frcp instruction is sometimes off by 1 ULP. For correctly
 * rounded frcp, a refinement step is required. This routine has been
 * exhaustively tested with a modified math_bruteforce.
 *
 * While Khronos APIs allow 2.5 ULP error for divides, nir_lower_idiv relies on
 * correctly rounded frcp. This is therefore load bearing for integer division
 * on all APIs.
 */
static nir_def *
libagx_frcp(nir_builder *b, nir_def *x)
{
   nir_def *u = nir_frcp(b, x);

   /* Do 1 Newton-Raphson refinement step.
    *
    * Define f(u) = xu - 1. Then f(u) = 0 iff u = 1/x. Newton's method gives:
    *
    * u_2 = u - f(u) / f'(u) = u - (xu - 1) / x
    *
    * Our original guess is close, so we approximate (1 / x) by u:
    *
    * u_2 = u - u(xu - 1) = u + u(1 - xu)
    *     = fma(fma(-x, u, 1), u, u)
    */
   nir_def *one = nir_imm_float(b, 1.0);
   nir_def *u_2 = nir_ffma(b, nir_ffma(b, nir_fneg(b, x), u, one), u, u);

   /* If the original value was infinite, frcp will generate the correct zero.
    * However, the Newton-Raphson step would multiply 0 * Inf and get a NaN. So
    * skip the refinement step for infinite inputs. We do this backwards,
    * checking whether the refined result is NaN, since we can implement this
    * check in a single fcmpsel instruction. The other case where the refinement
    * is NaN is a NaN input, in which skipping refinement is acceptable.
    */
   return nir_bcsel(b, nir_fisnan(b, u_2), u, u_2);
}

static bool
agx_nir_lower_fdiv(nir_builder *b, nir_alu_instr *alu, void *_)
{
   if (alu->op != nir_op_frcp || !nir_alu_instr_is_exact(alu) ||
       alu->def.bit_size != 32)
      return false;

   /* The refinement makes frcp correctly rounded, which only integer
    * division needs: nir_lower_idiv takes the reciprocal of an integer
    * converted to float. An application float division keeps the 2.5 ULP
    * that every Khronos API allows even when it is `precise`/NoContraction
    * (which forbids contraction, not the hardware reciprocal), so do not
    * spend two FMAs and a select on it; DXVK marks every operation precise.
    * Driver-internal shaders keep the refinement unconditionally. */
   if (!b->shader->info.internal) {
      nir_scalar src = nir_scalar_resolved(alu->src[0].src.ssa,
                                           alu->src[0].swizzle[0]);
      if (!nir_scalar_is_alu(src) ||
          (nir_scalar_alu_op(src) != nir_op_u2f32 &&
           nir_scalar_alu_op(src) != nir_op_i2f32))
         return false;
   }

   b->cursor = nir_before_instr(&alu->instr);
   nir_def_replace(&alu->def, libagx_frcp(b, nir_ssa_for_alu_src(b, alu, 0)));
   return true;
}

static bool
agx_nir_lower_g17p_f16_classify(nir_builder *b, nir_alu_instr *alu,
                                void *_data)
{
   if (alu->def.num_components != 1 ||
       nir_src_bit_size(alu->src[0].src) != 16)
      return false;

   const bool is_nan =
      alu->op == nir_op_fneu && nir_alu_srcs_equal(alu, alu, 0, 1);

   bool is_inf = false;
   nir_alu_instr *absolute = NULL;
   if (alu->op == nir_op_feq) {
      for (unsigned constant = 0; constant < 2; ++constant) {
         const unsigned value = constant ^ 1;
         nir_alu_instr *candidate = nir_src_as_alu(alu->src[value].src);
         /* The constant may be a vector read through a swizzle (isinf of a
          * vec2 compares each lane with one splatted infinity), so read the
          * lane this comparison uses rather than the whole source. */
         const nir_scalar k = nir_scalar_chase_alu_src(
            nir_get_scalar(&alu->def, 0), constant);
         const double compare =
            nir_scalar_is_const(k) ? nir_scalar_as_float(k) : 0.0;

         if (candidate && candidate->op == nir_op_fabs &&
             nir_src_bit_size(candidate->src[0].src) == 16 &&
             isinf(compare) && compare > 0.0) {
            is_inf = true;
            absolute = candidate;
         }
      }
   }

   if (!is_nan && !is_inf)
      return false;

   /* Classify the IEEE-754 binary16 encoding directly. Clearing the sign
    * leaves +infinity at 0x7c00 and every NaN strictly above it. This is exact
    * for quiet and signaling NaNs, both infinities, both zeros, subnormals,
    * and finite values. Ordinary half compares stay on their native path. */
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *value = is_nan ? nir_ssa_for_alu_src(b, alu, 0)
                           : nir_ssa_for_alu_src(b, absolute, 0);
   nir_def *magnitude =
      nir_iand_imm(b, nir_u2u32(b, value), UINT32_C(0x7fff));
   nir_def *result = is_nan
                        ? nir_ult(b, nir_imm_int(b, 0x7c00), magnitude)
                        : nir_ieq_imm(b, magnitude, 0x7c00);
   nir_def_replace(&alu->def, result);
   return true;
}

/* Shared preprocessing with target-bounded subgroup choices. */
static nir_mem_access_size_align
agx3_scratch_word_size_align_cb(nir_intrinsic_op intrin, uint8_t bytes,
                                uint8_t bit_size, uint32_t align,
                                uint32_t align_offset, bool offset_is_const,
                                enum gl_access_qualifier access,
                                const void *cb_data)
{
   return (nir_mem_access_size_align){
      .num_components = MIN2(MAX2(bytes / 4, 1), 4),
      .bit_size = 32,
      .align = 4,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

bool
agx3_nir_lower_scratch_words(nir_shader *nir)
{
   nir_lower_mem_access_bit_sizes_options scratch_words = {
      .modes = nir_var_function_temp,
      .callback = agx3_scratch_word_size_align_cb,
   };
   bool progress = false;
   NIR_PASS(progress, nir, nir_lower_mem_access_bit_sizes, &scratch_words);
   return progress;
}

static void
agx_preserve_quad_control(nir_shader *nir)
{
   /* G17P has one native helper-capable fragment entry protocol: the first
    * coefficient load leaves its input-start marker clear and the linked
    * program disables triangle merging. Fresh applegpu_g17p derivatives and
    * quad shuffles use that same protocol and carry otherwise identical
    * ProgramVariant metadata.
    *
    * nir_shader_gather_info derives these flags from individual operations,
    * but the quad-control execution modes are independently meaningful. In
    * particular, RequireFullQuadsKHR promises a full quad even when no quad
    * intrinsic survives optimization, and QuadDerivativesKHR requires the
    * derivative group to be the full quad. Preserve those source-level modes
    * explicitly instead of making the hardware protocol depend on whichever
    * intrinsics happen to remain at this point in lowering. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       (nir->info.fs.require_full_quads || nir->info.fs.quad_derivatives)) {
      nir->info.fs.needs_coarse_quad_helper_invocations = true;
      nir->info.fs.needs_full_quad_helper_invocations = true;
   }
}

static void
agx_preprocess_nir_internal(nir_shader *nir, bool g17p, bool native_fragment_depth,
                            bool g18g, bool m5, bool chip_g17p)
{
   if (!nir)
      return;

   if (g17p)
      agx_preserve_quad_control(nir);

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* Lower large arrays to scratch and small arrays to csel */
   NIR_PASS(_, nir, nir_lower_vars_to_scratch, AGX_SCRATCH_ARRAY_THRESHOLD_B,
            glsl_get_natural_size_align_bytes, glsl_get_word_size_align_bytes);

   /* The native stack protocol is grounded for 32-bit words and single
    * static halves only. Access scratch in whole words on AGX3: loads
    * over-read within the private frame and narrow stores become the
    * pass's read-modify-write of the containing word
    * (subgroups.ballot_broadcast.*.subgroupbroadcast_{f,i,u}16vec*,
    * {i,u}8vec*). */
   if (g17p)
      agx3_nir_lower_scratch_words(nir);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            nir_var_function_temp, ~0);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_var_copies);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* G18's native image/SSBO/atomic fragments feed pass-through depth
       * back after their side effects, just like G17P. The 1x and 4x
       * native318 reference matrix emits DEPTHF even without an explicit
       * shader depth output. The AGX2 all-live sample-mask lowering is not
       * that late-test contract. Keep helper-write suppression paired with
       * the native lowering; draw admission and multisample export remain
       * independently qualified by the driver. */
      NIR_PASS(_, nir, agx_nir_lower_frag_sidefx,
               g17p || native_fragment_depth || g18g);
      NIR_PASS(_, nir, nir_lower_is_helper_invocation);
   }

   /* Clean up deref gunk after lowering I/O */
   NIR_PASS(_, nir, nir_opt_dce);

   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   if (g17p || native_fragment_depth) {
      NIR_PASS(_, nir, nir_shader_alu_pass,
               agx_nir_lower_g17p_f16_classify,
               nir_metadata_control_flow, NULL);
   }
   NIR_PASS(_, nir, nir_lower_frexp);
   /* Before nir_lower_alu's generic lower_fminmax_signed_zero. Validated on
    * G17P only (`g17p` above means any native AGX3 chip); other chips keep
    * the generic lowering. The flush policy is agx_nir_lower_fminmax's. */
   if (chip_g17p) {
      /* DXVK and vkd3d-proton mark every float op signed-zero preserving.
       * Drop the flag where no use can tell -0 from +0, before the min/max
       * lowering pays for it. */
      NIR_PASS(_, nir, nir_opt_fp_math_ctrl);
      NIR_PASS(_, nir, agx_nir_lower_fminmax_signed_zero, g17p || m5);
   }
   NIR_PASS(_, nir, nir_lower_alu);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_flrp, 16 | 32 | 64, false);
   /* When the explicit-RTE FMA repair applies, G17P lowers sin/cos in
    * agx_compile_shader_nir after it, so the repair covers the shader's own
    * FMAs only. Other shaders keep the lowering here. */
   if (!g17p || !agx_nir_fma_rte_applies(nir))
      NIR_PASS(_, nir, agx_lower_sincos,
               g17p ? 1 : (native_fragment_depth || g18g) ? 2 : 0);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, agx_lower_front_face,
            nir_metadata_control_flow, NULL);
   if (g18g)
      NIR_PASS(_, nir, agx_nir_lower_g18_subgroups);
   if (g17p)
      NIR_PASS(_, nir, agx_nir_lower_subgroups_g17p);
   else if (m5)
      NIR_PASS(_, nir, agx_nir_lower_subgroups_m5, !g18g, !g18g);
   else
      NIR_PASS(_, nir, agx_nir_lower_subgroups);
   NIR_PASS(_, nir, nir_lower_all_phis_to_scalar);

   /* After lowering, run through the standard suite of NIR optimizations. We
    * will run through the loop later, once we have the shader key, but if we
    * run now, that run will ideally be almost a no-op.
    */
   agx_optimize_loop_nir(nir, !g17p && native_fragment_depth, g17p);

   /* Lower idiv after an optimization loop so we can constant fold more before
    * nir_opt_idiv_const.
    */
   NIR_PASS(_, nir, nir_opt_idiv_const, 16);

   nir_lower_idiv_options idiv_options = {
      .allow_fp16 = true,
   };

   NIR_PASS(_, nir, nir_lower_idiv, &idiv_options);

   /* Has to run after nir_lower_idiv */
   NIR_PASS(_, nir, nir_shader_alu_pass, agx_nir_lower_fdiv,
            nir_metadata_control_flow, NULL);

   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* We're lowered away all variables. Remove them all for smaller shaders. */
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_all, NULL);
   nir->info.io_lowered = true;

   /* Move before lowering */
   nir_move_options move_all = nir_move_const_undef | nir_move_load_ubo |
                               nir_move_load_input | nir_move_load_frag_coord |
                               nir_move_comparisons | nir_move_copies |
                               nir_move_load_ssbo | nir_move_alu;
   if (agx_has_derivative_in_control_flow(nir))
      move_all &= ~(nir_move_load_input | nir_move_load_frag_coord |
                    nir_move_alu);

   NIR_PASS(_, nir, nir_opt_sink, move_all);
   NIR_PASS(_, nir, nir_opt_move, move_all);
   /* Local-memory offsets are narrowed before the key-dependent optimizer so
    * conversion lowering and cleanup remain part of the normal pipeline.
    * G17P ordinary loads/stores use scalar-element indices; atomics retain
    * their byte-addressed encoding.
    */
   /* The conversion below scales the shared byte offset by the access width,
    * so any access that a later pass would split has to be split first.
    * agx_compile_shader_nir() runs nir_lower_mem_access_bit_sizes over
    * nir_var_mem_shared with mem_access_size_align_cb, which caps the width at
    * 32 bits; without this a 64-bit access is divided by 8 and only then split
    * into 32-bit halves, leaving each half addressing half its element.
    * Accesses of 32 bits and narrower are not split, so they are unaffected.
    */
   if (g17p || native_fragment_depth) {
      nir_lower_mem_access_bit_sizes_options shared_split = {
         .modes = nir_var_mem_shared,
         .callback = mem_access_size_align_cb,
      };
      NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &shared_split);
   }

   NIR_PASS(_, nir, agx_nir_lower_shared_bitsize,
            g17p || native_fragment_depth);

   /* A frontend that reaches this helper with `g17p` set owns the element-index
    * conversion for the whole shader, including shared accesses that do not
    * exist yet: hk_shader.c lowers `shared` variables to load_shared and
    * store_shared at :1810, which is *between* its two calls here (:252 and
    * :1892), so the first call legitimately finds nothing and the second does
    * the work.  Record the ownership rather than the work -- a flag set by the
    * pass itself would let that empty first call pass for a finished job and
    * suppress both the real conversion and the backstop below, which is
    * exactly how the Vulkan front end lost its conversion once already.
    */
   if (g17p || native_fragment_depth)
      nir->info.shared_element_index_agx = true;
}

void
agx_preprocess_nir(nir_shader *nir)
{
   agx_preprocess_nir_internal(nir, false, false, false, false, false);
}

void
agx_preprocess_nir_native(nir_shader *nir, enum agx_chip chip)
{
   agx_preprocess_nir_internal(nir, true, true, false, false,
                               chip == AGX_CHIP_G17P);
}

void
agx_preprocess_nir_g17p(nir_shader *nir)
{
   agx_preprocess_nir_native(nir, AGX_CHIP_G17P);
}

void
agx_preprocess_nir_g18g(nir_shader *nir)
{
   /* Select G18 trig without inheriting G17P subgroup/shared-memory rules. */
   agx_preprocess_nir_internal(nir, false, false, true, false, false);
}

static bool
lower_g17p_halfword_tile_load(nir_builder *b, nir_intrinsic_instr *intr,
                             UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_local_pixel_agx ||
       nir_intrinsic_format(intr) != PIPE_FORMAT_R16_UINT ||
       intr->def.bit_size != 32)
      return false;

   /* Native LLDCOL transfers tightly packed halfword lanes. AGX2 can
    * widen while loading, so its NIR may request W32 results from I16
    * storage. Load the physical lanes before zero-extending each one;
    * signed logical formats already extend in the tilebuffer lowering. */
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *raw = nir_load_local_pixel_agx(
      b, intr->def.num_components, 16, intr->src[0].ssa,
      .base = nir_intrinsic_base(intr), .format = PIPE_FORMAT_R16_UINT,
      .target = nir_intrinsic_target(intr));
   nir_def_replace(&intr->def, nir_u2u32(b, raw));
   return true;
}

static bool
lower_printf_buffer(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_device_key *dev = data;
   uint64_t value = 0;
   if (intr->intrinsic == nir_intrinsic_load_printf_buffer_address) {
      /* AGX3 frontends must publish the process-local printf address through
       * runtime state.  Baking it here would poison serialized shader caches;
       * the preflight below rejects an address intrinsic that escaped that
       * lowering.  AGX2 retains its established fixed-address ABI. */
      if (dev->encoding != AGX_ISA_ENCODING_AGX2)
         return false;
      value = LIBAGX_PRINTF_BUFFER_ADDRESS;
   } else if (intr->intrinsic == nir_intrinsic_load_printf_buffer_size) {
      value = LIBAGX_PRINTF_BUFFER_SIZE;
   } else {
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, value, intr->def.bit_size));
   return true;
}

/* A fragment shader that indexes a varying array at run time reads it through
 * one load_interpolated_input whose offset is a value rather than a constant.
 * That needs the interpolation to take its coefficient index from a register.
 * AGX2 encodes exactly that -- `iter r4_r5_r6_r7, r0l, cf0` -- but no AGX3B
 * form for it is witnessed, and Metal cannot express a dynamically indexed
 * interpolant, so metal-nt will not produce one either. Refusing the shader
 * is what KHR-GL33.shaders.indexing.varying_array's four *_dynamic_read cases
 * hit.
 *
 * Read every slot the array spans and select between them. This runs in the
 * compile rather than the preprocess because IO is lowered by then: the
 * indirection is an offset source, not a deref, so
 * nir_lower_indirect_derefs_to_if_else_trees cannot see it.
 *
 * It costs one interpolation per array element where AGX2 needs one in total.
 * That is an efficiency gap against the baseline, recorded as such -- not a
 * correctness one, and the alternative is refusing to compile at all.
 */
static bool
lower_indirect_fs_input(nir_builder *b, nir_intrinsic_instr *intr,
                        UNUSED void *data)
{
   unsigned offset_src;
   if (intr->intrinsic == nir_intrinsic_load_interpolated_input)
      offset_src = 1;
   else if (intr->intrinsic == nir_intrinsic_load_input)
      offset_src = 0;
   else
      return false;

   if (nir_src_is_const(intr->src[offset_src]))
      return false;

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   const unsigned slots = sem.num_slots;
   if (slots <= 1 || slots > 32)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *index = intr->src[offset_src].ssa;
   nir_def *result = NULL;

   for (unsigned i = 0; i < slots; ++i) {
      nir_io_semantics one_sem = sem;
      one_sem.num_slots = 1;
      one_sem.location = sem.location + i;

      nir_def *one;
      if (intr->intrinsic == nir_intrinsic_load_interpolated_input) {
         one = nir_load_interpolated_input(
            b, intr->def.num_components, intr->def.bit_size,
            intr->src[0].ssa, nir_imm_int(b, 0),
            .base = nir_intrinsic_base(intr) + i,
            .component = nir_intrinsic_component(intr),
            .dest_type = nir_intrinsic_dest_type(intr),
            .io_semantics = one_sem);
      } else {
         one = nir_load_input(
            b, intr->def.num_components, intr->def.bit_size,
            nir_imm_int(b, 0), .base = nir_intrinsic_base(intr) + i,
            .component = nir_intrinsic_component(intr),
            .dest_type = nir_intrinsic_dest_type(intr),
            .io_semantics = one_sem);
      }

      result = result == NULL
                  ? one
                  : nir_bcsel(b, nir_ieq_imm(b, index, i), one, result);
   }

   nir_def_replace(&intr->def, result);
   return true;
}

static bool
has_unlowered_printf_address(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_printf_buffer_address)
               return true;
         }
      }
   }

   return false;
}

/*
 * G17P uniform DMA (keyed by agx_shader_key::g17p_uniform_dma).
 *
 * Apple's G17P programs receive the constants whose address is known when
 * the launch is recorded -- push data, root and descriptor-set words, bound
 * buffer contents -- by DMA in the launch (Ioto) program, and their constant
 * program computes on those copies (ShaderInfo dma_list and
 * cprog_uniform_inputs). A G17P constant program runs once per batch, so a
 * device load left in it is paid per batch, while a copied range is not.
 *
 * Copy two kinds of constant-index loads:
 *
 *  - loads through a buffer descriptor the driver can read when it records
 *    the launch (a dynamic buffer in the root table, or a buffer in a static
 *    descriptor set). The driver zero-fills what lies past the buffer's end,
 *    so the per-component bounds checks of such loads are dropped, and the
 *    descriptor is usually left unused;
 *
 *  - loads through the root table or one descriptor set pointer: push data
 *    and descriptors still in use.
 *
 * Each source gets the window of at most AGX3_UNIFORM_DMA_MAX_HALFS covering
 * most of its loads; up to AGX3_UNIFORM_DMA_MAX_SPANS windows are placed in the
 * uniforms left after every reserved one and every constant-program result,
 * and each covered load becomes a uniform read. With a constant program, this
 * runs once it is formed: it then computes on the copies, and keeps the loads
 * for which no uniforms are left. Only raw 8/16/32-bit formats with a
 * constant, non-negative element index qualify.
 */
#define AGX3_UNIFORM_DMA_MAX_HALFS 128u

struct agx3_uniform_dma_candidate {
   nir_intrinsic_instr *load;
   uint8_t source;
   uint16_t pointer;
   uint16_t descriptor_B;
   uint32_t offset_B;
   uint32_t size_B;
};

static int
agx3_uniform_dma_compare(const void *a_, const void *b_)
{
   const struct agx3_uniform_dma_candidate *a = a_, *b = b_;
   if (a->source != b->source)
      return a->source < b->source ? -1 : 1;
   if (a->pointer != b->pointer)
      return a->pointer < b->pointer ? -1 : 1;
   if (a->descriptor_B != b->descriptor_B)
      return a->descriptor_B < b->descriptor_B ? -1 : 1;
   if (a->offset_B != b->offset_B)
      return a->offset_B < b->offset_B ? -1 : 1;
   return a->size_B < b->size_B ? -1 : a->size_B > b->size_B;
}

static bool
agx3_uniform_dma_same_source(const struct agx3_uniform_dma_candidate *a,
                             const struct agx3_uniform_dma_candidate *b)
{
   return a->source == b->source && a->pointer == b->pointer &&
          a->descriptor_B == b->descriptor_B;
}

/* The descriptor set whose pointer uniform is at `base`, -1 for the root
 * table, or -2 for any other uniform. */
static int
agx3_uniform_dma_pointer_set(const struct agx_shader_key *key, unsigned base)
{
   const unsigned root = key->g17p_dma_root;
   if (base == root)
      return -1;
   if (base < root + 4 || (base - root - 4) % 4 != 0 ||
       (base - root - 4) / 4 >= 8)
      return -2;
   return (base - root - 4) / 4;
}

/* The pointer uniform `ptr` loads, if it is the root table or a set in
 * set_mask, else -1. */
static int
agx3_uniform_dma_pointer(const struct agx_shader_key *key, nir_def *ptr,
                         uint8_t set_mask)
{
   nir_intrinsic_instr *intr = nir_def_as_intrinsic_or_null(ptr);
   if (!intr || intr->intrinsic != nir_intrinsic_load_preamble ||
       intr->def.bit_size != 64 || intr->def.num_components != 1)
      return -1;

   const unsigned base = nir_intrinsic_base(intr);
   const int set = agx3_uniform_dma_pointer_set(key, base);
   if (set == -2 || (set >= 0 && !(set_mask & BITFIELD_BIT(set))))
      return -1;
   return base;
}

/* Byte offset of a constant-index raw load_constant_agx, or -1. */
static int64_t
agx3_uniform_dma_offset(nir_intrinsic_instr *intr, unsigned *size_B)
{
   if (!nir_src_is_const(intr->src[1]))
      return -1;

   const unsigned field = nir_intrinsic_base(intr);
   if (field & (AGX_NIR_MEMORY_INDEX_BYTES | AGX_NIR_MEMORY_PACKED_COUNT_MASK))
      return -1;

   unsigned format_shift;
   const enum pipe_format format = nir_intrinsic_format(intr);
   if (format == PIPE_FORMAT_R32_UINT && intr->def.bit_size == 32)
      format_shift = 2;
   else if (format == PIPE_FORMAT_R16_UINT && intr->def.bit_size == 16)
      format_shift = 1;
   else if (format == PIPE_FORMAT_R8_UINT && intr->def.bit_size == 8)
      format_shift = 0;
   else
      return -1;

   const int64_t index = nir_intrinsic_sign_extend(intr)
                            ? nir_src_as_int(intr->src[1])
                            : (int64_t)nir_src_as_uint(intr->src[1]);
   if (index < 0 || index > (1 << 20) || field > 3)
      return -1;

   const uint64_t offset_B = (uint64_t)index << (format_shift + field);
   if (offset_B > (UINT32_C(1) << 24))
      return -1;

   *size_B = intr->def.num_components << format_shift;
   return offset_B;
}

/* Fill in a candidate whose address is a buffer descriptor's: the 64-bit
 * pack of two adjacent words of one constant-offset load through the root
 * table (a dynamic buffer descriptor) or a static set. */
static bool
agx3_uniform_dma_buffer(const struct agx_shader_key *key, nir_def *ptr,
                        struct agx3_uniform_dma_candidate *c)
{
   nir_alu_instr *pack = nir_def_as_alu_or_null(ptr);
   if (!pack || pack->op != nir_op_pack_64_2x32_split ||
       pack->src[0].src.ssa != pack->src[1].src.ssa ||
       pack->src[1].swizzle[0] != pack->src[0].swizzle[0] + 1)
      return false;

   nir_intrinsic_instr *desc = nir_src_as_intrinsic(pack->src[0].src);
   if (!desc || desc->intrinsic != nir_intrinsic_load_constant_agx ||
       desc->def.bit_size != 32)
      return false;

   const int pointer =
      agx3_uniform_dma_pointer(key, desc->src[0].ssa,
                               key->g17p_dma_static_set_mask);
   if (pointer < 0)
      return false;

   unsigned size_B;
   const int64_t offset_B = agx3_uniform_dma_offset(desc, &size_B);
   if (offset_B < 0)
      return false;

   /* A 16-byte buffer descriptor (address, then size): one of the root
    * table's, or a set's, where buffer descriptors are 16-byte aligned. */
   const int64_t descriptor_B = offset_B + 4 * pack->src[0].swizzle[0];
   const int64_t table_B = pointer == key->g17p_dma_root
                              ? descriptor_B - key->g17p_dma_buffers_B
                              : descriptor_B;
   if (table_B < 0 || table_B % 16 != 0 || descriptor_B > UINT16_MAX ||
       (pointer == key->g17p_dma_root &&
        table_B / 16 >= key->g17p_dma_buffer_count))
      return false;

   c->source = AGX3_UNIFORM_DMA_BUFFER;
   c->pointer = pointer;
   c->descriptor_B = descriptor_B;
   return true;
}

static void
agx3_uniform_dma_collect(nir_shader *nir, const struct agx_shader_key *key,
                         enum agx3_uniform_dma_source source,
                         struct util_dynarray *cands)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_constant_agx)
               continue;

            unsigned size_B;
            const int64_t offset_B = agx3_uniform_dma_offset(intr, &size_B);
            if (offset_B < 0)
               continue;

            struct agx3_uniform_dma_candidate c = {
               .load = intr,
               .source = AGX3_UNIFORM_DMA_POINTER,
               .offset_B = offset_B,
               .size_B = size_B,
            };

            if (source == AGX3_UNIFORM_DMA_POINTER) {
               const int pointer = agx3_uniform_dma_pointer(
                  key, intr->src[0].ssa, key->g17p_dma_set_mask);
               if (pointer < 0)
                  continue;
               c.pointer = pointer;
            } else if (!agx3_uniform_dma_buffer(key, intr->src[0].ssa, &c)) {
               continue;
            } else if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
                       nir->info.stage == MESA_SHADER_COMPUTE &&
                       intr->def.bit_size != 32) {
               /* M5 compute copies complete bounded dwords only. Preserve
                * narrow loads: a valid byte/half in an incomplete trailing
                * word cannot be represented by a whole-word zero tail. */
               continue;
            }
            util_dynarray_append(cands, c);
         }
      }
   }
}

struct agx3_uniform_dma_window {
   unsigned first, last, count;
   uint32_t start, end;
};

/* The window of at most AGX3_UNIFORM_DMA_MAX_HALFS covering the most of the
 * sorted candidates [first, end) of one source not yet taken. */
static struct agx3_uniform_dma_window
agx3_uniform_dma_best_window(const struct agx3_uniform_dma_candidate *c,
                             const bool *taken, unsigned first, unsigned end)
{
   struct agx3_uniform_dma_window best = {0};
   for (unsigned i = first; i < end; ++i) {
      if (taken[i])
         continue;

      struct agx3_uniform_dma_window w = {
         .first = i,
         .start = c[i].offset_B & ~3u,
         .end = c[i].offset_B & ~3u,
      };
      for (unsigned j = i; j < end; ++j) {
         if (taken[j])
            continue;
         const uint32_t e = ALIGN_POT(c[j].offset_B + c[j].size_B, 4);
         const uint32_t new_end = MAX2(w.end, e);
         if ((new_end - w.start) / 2 > AGX3_UNIFORM_DMA_MAX_HALFS)
            break;
         w.end = new_end;
         w.last = j;
         w.count++;
      }
      if (w.count > best.count ||
          (w.count == best.count && w.end - w.start < best.end - best.start))
         best = w;
   }
   return best;
}

/* Whether `size` is the size word of c's buffer descriptor: loaded through
 * the same pointer, 8 bytes after the address. */
static bool
agx3_uniform_dma_is_size(const struct agx3_uniform_dma_candidate *c,
                         nir_scalar size)
{
   size = nir_scalar_resolved(size.def, size.comp);
   nir_intrinsic_instr *load = nir_def_as_intrinsic_or_null(size.def);
   if (!load || load->intrinsic != nir_intrinsic_load_constant_agx ||
       load->def.bit_size != 32)
      return false;

   nir_intrinsic_instr *ptr = nir_src_as_intrinsic(load->src[0]);
   unsigned load_size_B;
   const int64_t offset_B = agx3_uniform_dma_offset(load, &load_size_B);
   return ptr && ptr->intrinsic == nir_intrinsic_load_preamble &&
          nir_intrinsic_base(ptr) == c->pointer && offset_B >= 0 &&
          offset_B + 4 * size.comp == c->descriptor_B + 8;
}

/* A buffer load's per-component bounds checks against its own descriptor's
 * size are exactly what the span's zero fill gives (see
 * AGX3_UNIFORM_DMA_BUFFER): drop those of the loads in `copied` (load ->
 * candidate). Other checks stay. */
static void
agx3_uniform_dma_drop_bounds(nir_shader *nir, struct hash_table *copied)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op != nir_op_bounds_agx ||
                !nir_src_is_const(alu->src[1].src))
               continue;

            nir_scalar data = nir_scalar_resolved(alu->src[0].src.ssa,
                                                  alu->src[0].swizzle[0]);
            nir_intrinsic_instr *load = nir_def_as_intrinsic_or_null(data.def);
            struct hash_entry *entry =
               load ? _mesa_hash_table_search(copied, load) : NULL;
            if (!entry)
               continue;

            const struct agx3_uniform_dma_candidate *c = entry->data;
            const unsigned comp_B = load->def.bit_size / 8;
            if (nir_alu_src_as_uint(alu->src[1]) !=
                   c->offset_B + (data.comp + 1) * comp_B ||
                !agx3_uniform_dma_is_size(
                   c, nir_get_scalar(alu->src[2].src.ssa,
                                     alu->src[2].swizzle[0])))
               continue;

            nir_builder b = nir_builder_at(nir_before_instr(instr));
            nir_def_replace(&alu->def, nir_channel(&b, &load->def, data.comp));
         }
      }
   }
}

static bool agx3_is_sample_handle(nir_def *def);

/* Copies sit at virtual uniform bases from here until the constant
 * program's results that remain are packed below them. */
#define AGX3_UNIFORM_DMA_VIRTUAL 1024u

/* A general constant-program result. */
struct agx3_uniform_dma_export {
   uint16_t base, size, align;
   /* Every read is a whole main-program read that is no sample handle, so it
    * can move to the copy the result duplicates. */
   bool forwardable;
   /* Every component duplicates a uniform or a candidate load. */
   bool copy;
   /* Components duplicating a candidate load not copied yet. */
   uint8_t pending;
};

/* Copying `load` copies one component of export `export`. */
struct agx3_uniform_dma_pair {
   nir_intrinsic_instr *load;
   unsigned export;
};

struct agx3_uniform_dma_layout {
   struct agx3_uniform_dma_export *exports;
   unsigned count;
   /* The results occupy [reserved, end); when compactable, every read in
    * there falls inside one result, so they can be packed again. */
   unsigned reserved, end;
   bool compactable;
};

static int
agx3_uniform_dma_export_at(const struct agx3_uniform_dma_layout *l,
                           unsigned half)
{
   for (unsigned i = 0; i < l->count; ++i) {
      if (half >= l->exports[i].base &&
          half < l->exports[i].base + l->exports[i].size)
         return i;
   }
   return -1;
}

static int
agx3_uniform_dma_export_compare(const void *a_, const void *b_)
{
   const struct agx3_uniform_dma_export *a = a_, *b = b_;
   return (int)a->base - (int)b->base;
}

/* Collect the general results of the constant program, ordered by base. */
static void
agx3_uniform_dma_layout_init(nir_shader *nir, unsigned reserved, unsigned end,
                             struct agx3_uniform_dma_layout *l)
{
   *l = (struct agx3_uniform_dma_layout){
      .exports = calloc(512, sizeof(struct agx3_uniform_dma_export)),
      .reserved = reserved,
      .end = end,
   };
   nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);
   if (!entrypoint->preamble || !l->exports)
      return;
   l->compactable = end <= 512;

   nir_foreach_block(block, entrypoint->preamble->impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         if (store->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(store) != nir_preamble_class_general)
            continue;

         const nir_def *value = store->src[0].ssa;
         const unsigned align = MAX2(value->bit_size, 16) / 16;
         const unsigned base = nir_intrinsic_base(store);
         const unsigned size = align * value->num_components;
         const int at = agx3_uniform_dma_export_at(l, base);
         if (base < reserved || base + size > end || l->count == 512 ||
             (at >= 0 && (l->exports[at].base != base ||
                          l->exports[at].size != size)) ||
             (at < 0 && agx3_uniform_dma_export_at(l, base + size - 1) >= 0)) {
            l->compactable = false;
            continue;
         }

         if (at >= 0) {
            /* Written again (one value per branch): keep it a result. */
            l->exports[at].forwardable = false;
            continue;
         }
         l->exports[l->count++] = (struct agx3_uniform_dma_export){
            .base = base,
            .size = size,
            .align = value->bit_size == 64 ? 4 : align,
            .forwardable = true,
         };
      }
   }

   nir_foreach_function_impl(impl, nir) {
      const bool in_preamble = impl == entrypoint->preamble->impl;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned base, size;
            bool whole_read = !in_preamble;
            if (intr->intrinsic == nir_intrinsic_load_preamble &&
                nir_intrinsic_preamble_class(intr) ==
                   nir_preamble_class_general) {
               base = nir_intrinsic_base(intr);
               size = MAX2(intr->def.bit_size, 16) / 16 *
                      intr->def.num_components;
               whole_read &= !agx3_is_sample_handle(&intr->def);
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
                       intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
               base = nir_intrinsic_desc_set(intr);
               size = 4;
               whole_read = false;
            } else {
               continue;
            }
            if (base < reserved || base >= end)
               continue;

            const int at = agx3_uniform_dma_export_at(l, base);
            if (at < 0 ||
                base + size > l->exports[at].base + l->exports[at].size) {
               l->compactable = false;
               continue;
            }
            l->exports[at].forwardable &=
               whole_read && base == l->exports[at].base &&
               size == l->exports[at].size;
         }
      }
   }

   qsort(l->exports, l->count, sizeof(*l->exports),
         agx3_uniform_dma_export_compare);
}

/* Where the results end once packed, without those freed[] says the copies
 * replace. */
static unsigned
agx3_uniform_dma_packed_end(const struct agx3_uniform_dma_layout *l,
                            const bool *freed)
{
   if (!l->compactable)
      return l->end;

   unsigned end = l->reserved;
   for (unsigned i = 0; i < l->count; ++i) {
      if (!freed[i])
         end = ALIGN_POT(end, l->exports[i].align) + l->exports[i].size;
   }
   return end;
}

/* Pack the results left after forwarding below the copies, in their order,
 * and return where they end. */
static unsigned
agx3_uniform_dma_compact(nir_shader *nir, unsigned reserved, unsigned end)
{
   struct agx3_uniform_dma_layout l;
   agx3_uniform_dma_layout_init(nir, reserved, end, &l);
   if (!l.compactable) {
      free(l.exports);
      return end;
   }

   uint16_t new_base[512];
   unsigned next = reserved;
   for (unsigned i = 0; i < l.count; ++i) {
      next = ALIGN_POT(next, l.exports[i].align);
      new_base[i] = next;
      next += l.exports[i].size;
   }

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const bool general =
               (intr->intrinsic == nir_intrinsic_load_preamble ||
                intr->intrinsic == nir_intrinsic_store_preamble) &&
               nir_intrinsic_preamble_class(intr) == nir_preamble_class_general;
            const bool bindless =
               intr->intrinsic == nir_intrinsic_bindless_image_agx ||
               intr->intrinsic == nir_intrinsic_bindless_sampler_agx;
            if (!general && !bindless)
               continue;

            const unsigned base = general ? nir_intrinsic_base(intr)
                                          : nir_intrinsic_desc_set(intr);
            if (base < reserved || base >= end)
               continue;
            const int at = agx3_uniform_dma_export_at(&l, base);
            const unsigned moved = new_base[at] + (base - l.exports[at].base);
            if (general)
               nir_intrinsic_set_base(intr, moved);
            else
               nir_intrinsic_set_desc_set(intr, moved);
         }
      }
   }

   free(l.exports);
   return next;
}

/* Record which results duplicate which candidate loads (`loads`: load ->
 * candidate), counting the components still to be copied. A component counts
 * as a copy of a buffer load through the bounds check copying drops. */
static void
agx3_uniform_dma_layout_copies(nir_shader *nir,
                               struct agx3_uniform_dma_layout *l,
                               struct hash_table *loads,
                               struct util_dynarray *pairs)
{
   nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);
   if (!entrypoint->preamble)
      return;

   for (unsigned i = 0; i < l->count; ++i)
      l->exports[i].copy = l->exports[i].forwardable;

   nir_foreach_block(block, entrypoint->preamble->impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         if (store->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(store) != nir_preamble_class_general)
            continue;

         const int at = agx3_uniform_dma_export_at(l, nir_intrinsic_base(store));
         if (at < 0 || !l->exports[at].copy)
            continue;

         struct agx3_uniform_dma_export *e = &l->exports[at];
         nir_def *value = store->src[0].ssa;
         const unsigned comp_B = value->bit_size / 8;
         const unsigned start = util_dynarray_num_elements(
            pairs, struct agx3_uniform_dma_pair);

         /* Forwarding needs the copied components contiguous: consecutive
          * uniform halves, or consecutive bytes of one copied source. */
         const struct agx3_uniform_dma_candidate *first = NULL;
         uint32_t first_B = 0;
         int first_half = -1;
         for (unsigned c = 0; c < value->num_components && e->copy; ++c) {
            nir_scalar s = nir_scalar_resolved(value, c);
            nir_alu_instr *bounds = nir_def_as_alu_or_null(s.def);
            if (bounds && bounds->op == nir_op_bounds_agx &&
                nir_src_is_const(bounds->src[1].src))
               s = nir_scalar_resolved(bounds->src[0].src.ssa,
                                       bounds->src[0].swizzle[0]);
            else
               bounds = NULL;

            nir_intrinsic_instr *load = nir_def_as_intrinsic_or_null(s.def);
            if (!load || load->def.bit_size != value->bit_size) {
               e->copy = false;
               break;
            }

            if (load->intrinsic == nir_intrinsic_load_preamble) {
               const int half = nir_intrinsic_base(load) +
                                s.comp * (comp_B / 2);
               e->copy = !bounds && !first &&
                         nir_intrinsic_preamble_class(load) ==
                            nir_preamble_class_general &&
                         (c == 0 ? (first_half = half) >= 0
                                 : half == first_half + (int)(c * comp_B / 2));
               continue;
            }

            struct hash_entry *entry = _mesa_hash_table_search(loads, load);
            const struct agx3_uniform_dma_candidate *cand =
               entry ? entry->data : NULL;
            const uint32_t offset_B = cand ? cand->offset_B + s.comp * comp_B : 0;
            e->copy =
               cand && first_half < 0 &&
               (c == 0 || (agx3_uniform_dma_same_source(cand, first) &&
                           offset_B == first_B + c * comp_B)) &&
               (!bounds ||
                (cand->source == AGX3_UNIFORM_DMA_BUFFER &&
                 nir_alu_src_as_uint(bounds->src[1]) == offset_B + comp_B &&
                 agx3_uniform_dma_is_size(
                    cand, nir_get_scalar(bounds->src[2].src.ssa,
                                         bounds->src[2].swizzle[0]))));
            if (e->copy) {
               if (c == 0) {
                  first = cand;
                  first_B = offset_B;
               }
               util_dynarray_append(pairs,
                  ((struct agx3_uniform_dma_pair){load, at}));
            }
         }

         if (!e->copy) {
            /* Not a copy after all: forget its pairs. */
            pairs->size = start * sizeof(struct agx3_uniform_dma_pair);
            continue;
         }
         e->pending = util_dynarray_num_elements(
            pairs, struct agx3_uniform_dma_pair) - start;
      }
   }
}

/* Place the best windows of `cands` while spans remain, most loads first,
 * at virtual bases from *vdest on. A window is taken only if, with the
 * results it lets go, the packed results and every copy still end within
 * `limit`. */
static void
agx3_uniform_dma_take(nir_shader *nir, const struct agx_shader_key *key,
                      struct agx_shader_info *info,
                      struct util_dynarray *cands, unsigned limit,
                      unsigned exports_end, unsigned *vdest, int *set_pointer)
{
   const unsigned n =
      util_dynarray_num_elements(cands, struct agx3_uniform_dma_candidate);
   struct agx3_uniform_dma_candidate *c = util_dynarray_begin(cands);
   if (!n)
      return;
   qsort(c, n, sizeof(*c), agx3_uniform_dma_compare);

   /* Sources as [first, end) runs of the sorted candidates. */
   struct {
      unsigned first, end;
   } groups[32];
   unsigned group_count = 0;
   for (unsigned g = 0; g < n && group_count < ARRAY_SIZE(groups);) {
      unsigned end = g;
      while (end < n && agx3_uniform_dma_same_source(&c[end], &c[g]))
         ++end;
      groups[group_count].first = g;
      groups[group_count++].end = end;
      g = end;
   }

   struct hash_table *index = _mesa_pointer_hash_table_create(NULL);
   for (unsigned k = 0; k < n; ++k)
      _mesa_hash_table_insert(index, c[k].load, &c[k]);

   struct agx3_uniform_dma_layout l;
   struct util_dynarray pairs = UTIL_DYNARRAY_INIT;
   agx3_uniform_dma_layout_init(nir, key->reserved_preamble, exports_end, &l);
   agx3_uniform_dma_layout_copies(nir, &l, index, &pairs);
   const unsigned pair_count =
      util_dynarray_num_elements(&pairs, struct agx3_uniform_dma_pair);
   const struct agx3_uniform_dma_pair *pair = util_dynarray_begin(&pairs);

   bool *taken = calloc(n, sizeof(bool));
   bool freed[512] = {false};
   uint8_t pending[512];
   for (unsigned i = 0; i < l.count; ++i) {
      pending[i] = l.exports[i].pending;
      freed[i] = l.exports[i].copy && !pending[i];
   }

   while (info->agx3_uniform_dma_count < AGX3_UNIFORM_DMA_MAX_SPANS) {
      /* Windows of the sources, most loads first. The driver keys a
       * published launch on at most one descriptor set pointer value. */
      struct agx3_uniform_dma_window windows[ARRAY_SIZE(groups)];
      unsigned window_count = 0;
      for (unsigned g = 0; g < group_count; ++g) {
         const struct agx3_uniform_dma_candidate *head = &c[groups[g].first];
         if (head->source == AGX3_UNIFORM_DMA_POINTER &&
             head->pointer != key->g17p_dma_root && *set_pointer >= 0 &&
             head->pointer != *set_pointer)
            continue;
         struct agx3_uniform_dma_window w = agx3_uniform_dma_best_window(
            c, taken, groups[g].first, groups[g].end);
         if (w.count)
            windows[window_count++] = w;
      }

      int pick = -1;
      bool pick_freed[512];
      while (pick < 0 && window_count) {
         unsigned best = 0;
         for (unsigned i = 1; i < window_count; ++i) {
            if (windows[i].count > windows[best].count)
               best = i;
         }
         const struct agx3_uniform_dma_window *w = &windows[best];

         /* Results this window would let go: those whose every copied
          * component it holds, so their copies stay contiguous. */
         uint8_t left[512], held[512] = {0};
         memcpy(left, pending, l.count);
         memcpy(pick_freed, freed, sizeof(freed));
         for (unsigned p = 0; p < pair_count; ++p) {
            struct hash_entry *entry =
               _mesa_hash_table_search(index, pair[p].load);
            const unsigned k =
               (const struct agx3_uniform_dma_candidate *)entry->data - c;
            const unsigned e = pair[p].export;
            if (k < w->first || k > w->last || taken[k])
               continue;
            ++held[e];
            if (!--left[e] && held[e] == l.exports[e].pending)
               pick_freed[e] = l.exports[e].copy;
         }

         const unsigned halfs = (w->end - w->start) / 2;
         const unsigned copies = *vdest - AGX3_UNIFORM_DMA_VIRTUAL;
         if (ALIGN_POT(agx3_uniform_dma_packed_end(&l, pick_freed), 4) +
                copies + halfs <= limit) {
            pick = best;
            memcpy(pending, left, l.count);
         } else {
            windows[best] = windows[--window_count];
         }
      }
      if (pick < 0)
         break;

      const struct agx3_uniform_dma_window *w = &windows[pick];
      const unsigned halfs = (w->end - w->start) / 2;
      memcpy(freed, pick_freed, sizeof(freed));

      const struct agx3_uniform_dma_candidate *head = &c[w->first];
      if (head->source == AGX3_UNIFORM_DMA_POINTER &&
          head->pointer != key->g17p_dma_root)
         *set_pointer = head->pointer;

      if (head->source == AGX3_UNIFORM_DMA_BUFFER) {
         struct hash_table *copied = _mesa_pointer_hash_table_create(NULL);
         for (unsigned k = w->first; k <= w->last; ++k) {
            if (!taken[k])
               _mesa_hash_table_insert(copied, c[k].load, &c[k]);
         }
         agx3_uniform_dma_drop_bounds(nir, copied);
         _mesa_hash_table_destroy(copied, NULL);
      }

      for (unsigned k = w->first; k <= w->last; ++k) {
         if (taken[k])
            continue;
         taken[k] = true;

         nir_intrinsic_instr *load = c[k].load;
         nir_builder b = nir_builder_at(nir_before_instr(&load->instr));
         const unsigned offset_B = *vdest * 2 + c[k].offset_B - w->start;
         nir_def *value;
         if (load->def.bit_size == 8) {
            /* Bytes come out of their copied dwords. */
            nir_def *bytes[NIR_MAX_VEC_COMPONENTS];
            for (unsigned i = 0; i < load->def.num_components; ++i) {
               const unsigned byte_B = offset_B + i;
               nir_def *dword = nir_load_preamble(&b, 1, 32,
                                                  .base = (byte_B & ~3u) / 2);
               bytes[i] = nir_u2u8(&b, nir_ushr_imm(&b, dword,
                                                    8 * (byte_B & 3)));
            }
            value = nir_vec(&b, bytes, load->def.num_components);
         } else {
            value = nir_load_preamble(&b, load->def.num_components,
                                      load->def.bit_size,
                                      .base = offset_B / 2);
         }
         nir_def_replace(&load->def, value);
      }

      info->agx3_uniform_dma[info->agx3_uniform_dma_count++] =
         (struct agx3_uniform_dma_span){
            .source = head->source,
            .pointer = head->pointer,
            .descriptor_B = head->descriptor_B,
            .dest = *vdest,
            .halfs = halfs,
            .offset_B = w->start,
         };
      *vdest += ALIGN_POT(halfs, 4);
   }

   free(taken);
   free(l.exports);
   util_dynarray_fini(&pairs);
   _mesa_hash_table_destroy(index, NULL);
}

/* Whether a sample names `def` as its texture or sampler handle. */
static bool
agx3_is_sample_handle(nir_def *def)
{
   nir_foreach_use(use, def) {
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->type != nir_instr_type_tex)
         continue;
      const nir_tex_src *src = (const nir_tex_src *)use;
      if (src->src_type == nir_tex_src_texture_handle ||
          src->src_type == nir_tex_src_sampler_handle ||
          src->src_type == nir_tex_src_texture_offset ||
          src->src_type == nir_tex_src_sampler_offset)
         return true;
   }
   return false;
}

/* A constant-program result that only copies a uniform the launch already
 * holds is read straight from that uniform, and is no longer stored when
 * nothing else reads it. A sample handle stays a result when the copy is
 * above the halves a sample reads directly, so it can be placed there
 * (agx_nir_place_preamble_handles_low). */
static void
agx3_uniform_dma_forward_copies(nir_shader *nir)
{
   if (!nir_shader_get_entrypoint(nir)->preamble)
      return;
   nir_function_impl *preamble = nir_shader_get_preamble(nir);

   /* For each result half: the uniform half it copies, or -1. */
   int16_t source[512];
   memset(source, 0xff, sizeof(source));
   bool any = false;
   nir_foreach_block(block, preamble) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         if (store->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(store) != nir_preamble_class_general)
            continue;

         /* Each component a uniform half (pair) through movs and vecs. */
         nir_def *value = store->src[0].ssa;
         const unsigned comp_halfs = value->bit_size / 16;
         const unsigned base = nir_intrinsic_base(store);
         if (!comp_halfs || base + comp_halfs * value->num_components >
                               ARRAY_SIZE(source))
            continue;

         int16_t halves[64];
         bool copy = comp_halfs * value->num_components <= ARRAY_SIZE(halves);
         for (unsigned c = 0; c < value->num_components && copy; ++c) {
            nir_scalar scalar = nir_scalar_resolved(value, c);
            nir_intrinsic_instr *load =
               nir_def_as_intrinsic_or_null(scalar.def);
            copy = load && load->intrinsic == nir_intrinsic_load_preamble &&
                   nir_intrinsic_preamble_class(load) ==
                      nir_preamble_class_general &&
                   load->def.bit_size == value->bit_size;
            for (unsigned h = 0; copy && h < comp_halfs; ++h) {
               halves[c * comp_halfs + h] =
                  nir_intrinsic_base(load) + scalar.comp * comp_halfs + h;
            }
         }
         if (!copy)
            continue;
         for (unsigned h = 0; h < comp_halfs * value->num_components; ++h)
            source[base + h] = halves[h];
         any = true;
      }
   }
   if (!any)
      return;

   /* Forward main-program reads of whole copies; any other read of a copied
    * half keeps its store. */
   bool keep[512] = {false};
   nir_foreach_function_impl(impl, nir) {
      const bool in_preamble = impl == preamble;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            unsigned base, size;
            if (intr->intrinsic == nir_intrinsic_load_preamble &&
                nir_intrinsic_preamble_class(intr) ==
                   nir_preamble_class_general) {
               base = nir_intrinsic_base(intr);
               size = MAX2(intr->def.bit_size, 16) / 16 *
                      intr->def.num_components;
            } else if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
                       intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
               base = nir_intrinsic_desc_set(intr);
               size = 4;
               intr = NULL;
            } else {
               continue;
            }
            if (base + size > ARRAY_SIZE(source))
               continue;

            bool whole = !in_preamble && intr != NULL && source[base] >= 0;
            for (unsigned h = 1; h < size; ++h)
               whole &= source[base + h] == source[base] + h;
            whole &= !(intr && source[base] + size > AGX3_SMP_HANDLE_MAX_HALF &&
                       agx3_is_sample_handle(&intr->def));

            if (whole)
               nir_intrinsic_set_base(intr, source[base]);
            else
               memset(keep + base, true, size);
         }
      }
   }

   nir_foreach_block(block, preamble) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         if (store->intrinsic != nir_intrinsic_store_preamble ||
             nir_intrinsic_preamble_class(store) != nir_preamble_class_general)
            continue;

         const unsigned base = nir_intrinsic_base(store);
         const unsigned size = MAX2(store->src[0].ssa->bit_size, 16) / 16 *
                               store->src[0].ssa->num_components;
         if (base + size > ARRAY_SIZE(source) || source[base] < 0)
            continue;

         bool kept = false;
         for (unsigned h = 0; h < size; ++h)
            kept |= keep[base + h];
         if (!kept)
            nir_instr_remove(instr);
      }
   }
}

/* Uniform halves to leave for the main program's promoted constants
 * (agx_opt_promote_constants), which are placed after the spans: about one
 * per distinct constant too wide to be an immediate. */
static unsigned
agx3_uniform_dma_constant_reserve(nir_shader *nir)
{
   struct set *values = _mesa_pointer_set_create(NULL);
   unsigned halves = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_load_const)
            continue;
         nir_load_const_instr *load = nir_instr_as_load_const(instr);
         if (load->def.bit_size < 16)
            continue;
         for (unsigned c = 0; c < load->def.num_components; ++c) {
            const uint64_t value = nir_const_value_as_uint(load->value[c],
                                                           load->def.bit_size);
            /* Keyed by value and size; an estimate tolerates aliasing. */
            const void *key =
               (const void *)(uintptr_t)((value << 7) | load->def.bit_size);
            if (value <= 0xff || _mesa_set_search(values, key))
               continue;
            _mesa_set_add(values, key);
            halves += load->def.bit_size / 16;
         }
      }
   }
   _mesa_set_destroy(values, NULL);
   return MIN2(halves, 128);
}

/* Whether a vertex constant program only moves loaded data to its results:
 * every instruction is a load (of the preamble's inputs or of constant
 * memory), a constant, a result store, or plumbing that forwards values --
 * moves, vectors, 64-bit packs, address masks and bounds checks -- apart
 * from at most four other operations, which move into the main program.
 *
 * Copying such a program's data into the launch removes the program: its
 * results become the copies (agx3_uniform_dma_forward_copies()). A program
 * that also computes keeps running per vertex batch, and copying its raw
 * inputs only adds them to the uniforms the vertex program declares -- each
 * declared half costs per vertex batch too (hw/dbench d6: the same vertex
 * program with 44 vs 116 uniform halves, 2.72 vs 3.4-3.6 ms). In Minecraft,
 * whose vertex constant programs compute ProjMat * ModelViewMat, copying
 * doubled the vertex uniforms and cost 0.42 ms of GPU time per frame. */
static bool
agx3_vertex_preamble_only_forwards(nir_shader *nir)
{
   /* As many other operations as agx3_inline_cheap_vertex_preamble() moves
    * into the main program once the copies leave nothing else. */
   unsigned other = 0;
   nir_foreach_function_impl(impl, nir) {
      if (!impl->function->is_preamble)
         continue;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            switch (instr->type) {
            case nir_instr_type_load_const:
            case nir_instr_type_undef:
               continue;
            case nir_instr_type_intrinsic: {
               const nir_intrinsic_op op =
                  nir_instr_as_intrinsic(instr)->intrinsic;
               if (op == nir_intrinsic_load_preamble ||
                   op == nir_intrinsic_store_preamble ||
                   op == nir_intrinsic_load_constant_agx)
                  continue;
               return false;
            }
            case nir_instr_type_alu:
               switch (nir_instr_as_alu(instr)->op) {
               case nir_op_mov:
               case nir_op_vec2:
               case nir_op_vec3:
               case nir_op_vec4:
               case nir_op_pack_64_2x32_split:
               case nir_op_unpack_64_2x32_split_x:
               case nir_op_unpack_64_2x32_split_y:
               case nir_op_iand:
               case nir_op_bounds_agx:
                  continue;
               default:
                  if (++other > 4)
                     return false;
                  continue;
               }
            default:
               return false;
            }
         }
      }
   }
   return true;
}

bool
agx3_nir_lower_uniform_dma(nir_shader *nir, const struct agx_shader_key *key,
                           struct agx_shader_info *info)
{
   /* A vertex constant program that computes keeps its loads. */
   if (nir->info.stage == MESA_SHADER_VERTEX &&
       !agx3_vertex_preamble_only_forwards(nir))
      return false;

   const unsigned exports_end = info->push_count;
   unsigned vdest = AGX3_UNIFORM_DMA_VIRTUAL;
   int set_pointer = -1;
   info->agx3_uniform_dma_count = 0;

   /* Copies end where the constant program's storage does, before the main
    * program's promoted constants, and a fragment shader's uniforms end by
    * AGX3_FRAGMENT_UNIFORM_END. */
   const unsigned reserve = agx3_uniform_dma_constant_reserve(nir);
   const unsigned limit =
      nir->info.stage == MESA_SHADER_FRAGMENT
         ? AGX3_FRAGMENT_UNIFORM_END - MIN2(reserve, AGX3_FRAGMENT_UNIFORM_END)
         : MIN2(480, AGX_NUM_UNIFORMS - reserve);

   /* Buffers first: copying one also retires its descriptor's loads. */
   struct util_dynarray cands = UTIL_DYNARRAY_INIT;
   agx3_uniform_dma_collect(nir, key, AGX3_UNIFORM_DMA_BUFFER, &cands);
   agx3_uniform_dma_take(nir, key, info, &cands, limit, exports_end, &vdest,
                         &set_pointer);

   if (info->agx3_uniform_dma_count)
      nir_opt_dce(nir);

   util_dynarray_clear(&cands);
   agx3_uniform_dma_collect(nir, key, AGX3_UNIFORM_DMA_POINTER, &cands);
   agx3_uniform_dma_take(nir, key, info, &cands, limit, exports_end, &vdest,
                         &set_pointer);
   util_dynarray_fini(&cands);

   if (!info->agx3_uniform_dma_count)
      return false;

   /* Drop the results the copies duplicate, pack the rest, and place the
    * copies right above them. */
   agx3_uniform_dma_forward_copies(nir);
   const unsigned start = ALIGN_POT(
      agx3_uniform_dma_compact(nir, key->reserved_preamble, exports_end), 4);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_preamble &&
                nir_intrinsic_base(intr) >= AGX3_UNIFORM_DMA_VIRTUAL)
               nir_intrinsic_set_base(intr, nir_intrinsic_base(intr) -
                                               AGX3_UNIFORM_DMA_VIRTUAL + start);
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   for (unsigned i = 0; i < info->agx3_uniform_dma_count; ++i)
      info->agx3_uniform_dma[i].dest += start - AGX3_UNIFORM_DMA_VIRTUAL;
   info->push_count = start + vdest - AGX3_UNIFORM_DMA_VIRTUAL;
   assert(info->push_count <= AGX_NUM_UNIFORMS);
   return true;
}

/* AGX_G17P_DISCARD_WORD.
 *
 * At one sample the AGX3B coverage update reads only bit 0 of TARGET and LIVE
 * (the sample_mask_agx case builds target = T & 1 and keep = L & 1). The NIR
 * lowering builds both from whole-pixel masks, so a Minecraft-style
 * `if (a == 0) discard;` reaches the backend as
 *
 *    sample_mask_agx(tests_zs | bcsel(c, 0xff, 0), bcsel(c, 0, 0xff))
 *
 * and then two ANDs, two padded collects and a 32x32 IMAD to select the word
 * (Apple's native conditional discard is one FSEL and two IADDs). Rebuild each
 * non-constant source from its bit 0 -- selects of constants carry 0/1
 * directly and the uniform `tests_zs & 1` is hoisted into the preamble -- and
 * record that in FLAGS (bit i: source i is already 0 or 1), so the backend
 * drops its ANDs and selects with one widening IMAD. Constant sources are
 * left untouched, so the backend's no-tests analysis sees exactly what it saw
 * before. The value of bit 0 -- the only bit the one-sample path reads -- is
 * unchanged by construction.
 */
static bool
agx_g17p_bit0_const(nir_def *d, unsigned *v)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(d, 0));
   if (!nir_scalar_is_const(s))
      return false;
   *v = nir_scalar_as_uint(s) & 1;
   return true;
}

/* d = bcsel(c, #x, #y) with 0/1 constant arms, as built below. */
static bool
agx_g17p_bit0_const_select(nir_def *d, nir_def **c, unsigned *x, unsigned *y)
{
   nir_scalar s = nir_get_scalar(d, 0);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_bcsel)
      return false;
   nir_alu_instr *alu = nir_def_as_alu(d);
   if (!agx_g17p_bit0_const(alu->src[1].src.ssa, x) ||
       !agx_g17p_bit0_const(alu->src[2].src.ssa, y) ||
       alu->src[0].src.ssa->num_components != 1)
      return false;
   *c = alu->src[0].src.ssa;
   return true;
}

static nir_def *agx_g17p_bit0(nir_builder *b, nir_scalar s, unsigned depth);

static nir_def *
agx_g17p_bit0_logic(nir_builder *b, nir_op op, nir_def *l, nir_def *r)
{
   /* Fold constants here: a source that only reduces to a constant must be
    * recognised as one, so it keeps its original form (no-tests analysis). */
   unsigned lv, rv;
   if (agx_g17p_bit0_const(l, &lv) && agx_g17p_bit0_const(r, &rv)) {
      return nir_imm_intN_t(b, op == nir_op_ior    ? (lv | rv)
                               : op == nir_op_iand ? (lv & rv)
                                                   : (lv ^ rv), 16);
   }
   for (unsigned i = 0; i < 2; ++i) {
      nir_def *k = i ? r : l, *o = i ? l : r;
      unsigned kv, x, y;
      nir_def *c;
      if (!agx_g17p_bit0_const(k, &kv))
         continue;
      if (op == nir_op_ior)
         return kv ? nir_imm_intN_t(b, 1, 16) : o;
      if (op == nir_op_iand)
         return kv ? o : nir_imm_intN_t(b, 0, 16);
      if (!kv)
         return o;
      /* bcsel(c, #x, #y) ^ 1 = bcsel(c, #!x, #!y): live = killed ^ 0xff. */
      if (agx_g17p_bit0_const_select(o, &c, &x, &y))
         return nir_bcsel(b, c, nir_imm_intN_t(b, !x, 16),
                          nir_imm_intN_t(b, !y, 16));
      return nir_ixor(b, o, nir_imm_intN_t(b, 1, 16));
   }
   /* Leave `x | bcsel(c, 1, 0)` as an OR: distributing it would give a
    * select between a register or uniform and an immediate, which the AGX3B
    * compare-select materializes with extra moves. */
   return nir_build_alu2(b, op, l, r);
}

static nir_def *
agx_g17p_bit0(nir_builder *b, nir_scalar s, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s))
      return nir_imm_intN_t(b, nir_scalar_as_uint(s) & 1, 16);

   if (depth < 6 && nir_scalar_is_alu(s)) {
      const nir_op op = nir_scalar_alu_op(s);
      switch (op) {
      case nir_op_u2u16:
      case nir_op_u2u32:
      case nir_op_i2i16:
      case nir_op_i2i32:
         /* Integer conversions keep bit 0. */
         return agx_g17p_bit0(b, nir_scalar_chase_alu_src(s, 0), depth + 1);
      case nir_op_b2i16:
      case nir_op_b2i32: {
         nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         if (nir_scalar_is_const(c))
            return nir_imm_intN_t(b, nir_scalar_as_bool(c) ? 1 : 0, 16);
         return nir_b2i16(b, nir_channel(b, c.def, c.comp));
      }
      case nir_op_bcsel: {
         nir_scalar cs = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         if (nir_scalar_is_const(cs))
            return agx_g17p_bit0(
               b, nir_scalar_chase_alu_src(s, nir_scalar_as_bool(cs) ? 1 : 2),
               depth + 1);
         nir_def *c = nir_channel(b, cs.def, cs.comp);
         nir_def *x = agx_g17p_bit0(b, nir_scalar_chase_alu_src(s, 1), depth + 1);
         nir_def *y = agx_g17p_bit0(b, nir_scalar_chase_alu_src(s, 2), depth + 1);
         unsigned xv, yv;
         if (agx_g17p_bit0_const(x, &xv) && agx_g17p_bit0_const(y, &yv) &&
             xv == yv)
            return x;
         return nir_bcsel(b, c, x, y);
      }
      case nir_op_ior:
      case nir_op_iand:
      case nir_op_ixor:
         return agx_g17p_bit0_logic(
            b, op, agx_g17p_bit0(b, nir_scalar_chase_alu_src(s, 0), depth + 1),
            agx_g17p_bit0(b, nir_scalar_chase_alu_src(s, 1), depth + 1));
      default:
         break;
      }
   }

   nir_def *d = nir_channel(b, s.def, s.comp);
   if (d->bit_size == 1)
      return nir_b2i16(b, d);
   if (d->bit_size != 16)
      d = nir_u2u16(b, d);
   return nir_iand_imm(b, d, 1);
}

static bool
agx_g17p_discard_word_instr(nir_builder *b, nir_intrinsic_instr *intr,
                            void *data)
{
   if (intr->intrinsic != nir_intrinsic_sample_mask_agx ||
       nir_intrinsic_flags(intr))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   unsigned flags = 0;
   for (unsigned i = 0; i < 2; ++i) {
      if (nir_src_is_const(intr->src[i]))
         continue;
      nir_def *bit0 = agx_g17p_bit0(b, nir_get_scalar(intr->src[i].ssa, 0), 0);
      unsigned v;
      /* A source that reduces to a constant keeps its original form; the
       * no-tests decision reads constant sources. */
      if (agx_g17p_bit0_const(bit0, &v))
         continue;
      nir_src_rewrite(&intr->src[i], bit0);
      flags |= BITFIELD_BIT(i);
   }
   nir_intrinsic_set_flags(intr, flags);
   return flags != 0;
}

static bool
agx_g17p_discard_word(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, agx_g17p_discard_word_instr,
                                     nir_metadata_control_flow, NULL);
}

/* AGX_M5_DISCARD_WORD.
 *
 * M5 keeps a whole-pixel coverage update as one testing MOVMSK whose word is
 * SR2 repeated in the low two bytes and the live bit in byte 3 (the
 * sample_mask_agx case). For Minecraft's `if (color.a < 0.5) discard;` the
 * backend received sample_mask_agx(0xff, bcsel(c, 0, 0xff)) and spent an AND,
 * two shifting IADDs and a select per fragment on the live byte, plus an AND
 * and a select on the constant target. Apple's native conditional discard
 * selects byte 3's half directly (one FSEL). With a constant target naming
 * the one sample, rebuild LIVE as its bit 0 shifted to bit 8 of the high half
 * -- selects of constants keep one select, now of shifted constants -- and
 * record that in FLAGS bit 2, so the backend collects it as the high half
 * unchanged. Bit 0 is the only bit the one-sample word reads, so the update
 * is unchanged by construction.
 */
/* Whether component d_comp of d is b2f of component c_comp of c or its
 * inversion. Keep the consumer and producer swizzles composed. */
static bool
agx_b2f_of(nir_def *d, unsigned d_comp, nir_def *c, unsigned c_comp,
           bool *inverted)
{
   nir_alu_instr *b2f = nir_def_as_alu_or_null(d);
   if (!b2f || (b2f->op != nir_op_b2f32 && b2f->op != nir_op_b2f16))
      return false;
   nir_def *x = b2f->src[0].src.ssa;
   unsigned xc = b2f->src[0].swizzle[d_comp];
   nir_alu_instr *inot = nir_def_as_alu_or_null(x);
   *inverted = inot && inot->op == nir_op_inot;
   if (*inverted) {
      x = inot->src[0].src.ssa;
      xc = inot->src[0].swizzle[xc];
   }
   return x == c && xc == c_comp;
}

/*
 * bcsel(a || b, b2f(!a), r) is 0.0 when a holds, else 1.0 when b holds, else
 * r: split it into bcsel(a, 0.0, bcsel(b, 1.0, r)) (likewise for b2f(a) and
 * for b in place of a). Minecraft's linear fog builds this per distance (0
 * before the start, 1 past the end, the ramp between); AGX has no boolean
 * OR or NOT, so the backend built it from three selects, two of them needing
 * their uniform comparand copied into a GPR on M5. Apple's fragment uses two
 * selects. The values are exact (b2f gives +0.0 and 1.0). M5 only, so other
 * targets keep their compiles.
 */
static bool
agx_m5_split_or_select(nir_builder *b, nir_alu_instr *sel, void *data)
{
   if (sel->op != nir_op_bcsel || sel->def.num_components != 1)
      return false;
   nir_alu_instr *ior = nir_def_as_alu_or_null(sel->src[0].src.ssa);
   if (!ior || ior->op != nir_op_ior || ior->def.num_components != 1 ||
       ior->def.bit_size != 1)
      return false;

   for (unsigned first = 0; first < 2; ++first) {
      nir_def *a = ior->src[first].src.ssa;
      unsigned ac = ior->src[first].swizzle[0];
      bool inverted;
      if (!agx_b2f_of(sel->src[1].src.ssa, sel->src[1].swizzle[0],
                      a, ac, &inverted) ||
          a->num_components <= ac)
         continue;

      b->cursor = nir_before_instr(&sel->instr);
      const unsigned bits = sel->def.bit_size;
      nir_def *ca = nir_channel(b, a, ac);
      nir_def *cb = nir_mov_alu(b, ior->src[!first], 1);
      nir_def *r = nir_mov_alu(b, sel->src[2], 1);
      /* a holds: b2f(a) = 1, b2f(!a) = 0. Otherwise b holds: the opposite. */
      nir_def *when_a = nir_imm_floatN_t(b, inverted ? 0.0 : 1.0, bits);
      nir_def *when_b = nir_imm_floatN_t(b, inverted ? 1.0 : 0.0, bits);
      nir_def *split = nir_bcsel(b, ca, when_a, nir_bcsel(b, cb, when_b, r));
      nir_def_replace(&sel->def, split);
      return true;
   }
   return false;
}

bool
agx_nir_m5_split_or_selects(nir_shader *nir)
{
   return nir_shader_alu_pass(nir, agx_m5_split_or_select,
                              nir_metadata_control_flow, NULL);
}

static bool
agx_m5_discard_word_instr(nir_builder *b, nir_intrinsic_instr *intr,
                          void *data)
{
   if (intr->intrinsic != nir_intrinsic_sample_mask_agx ||
       nir_intrinsic_flags(intr) || !nir_src_is_const(intr->src[0]) ||
       !(nir_src_as_uint(intr->src[0]) & 1) || nir_src_is_const(intr->src[1]))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *bit0 = agx_g17p_bit0(b, nir_get_scalar(intr->src[1].ssa, 0), 0);
   unsigned v, x, y;
   nir_def *c;
   /* A live mask that reduces to a constant keeps its original form; the
    * no-tests decision reads constant sources. */
   if (agx_g17p_bit0_const(bit0, &v))
      return false;
   nir_def *high = agx_g17p_bit0_const_select(bit0, &c, &x, &y)
      ? nir_bcsel(b, c, nir_imm_intN_t(b, x << 8, 16),
                  nir_imm_intN_t(b, y << 8, 16))
      : nir_ishl_imm(b, bit0, 8);
   nir_src_rewrite(&intr->src[1], high);
   nir_intrinsic_set_flags(intr, BITFIELD_BIT(2));
   return true;
}

static bool
agx_m5_discard_word(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, agx_m5_discard_word_instr,
                                     nir_metadata_control_flow, NULL);
}

unsigned
agx_max_vertex_attribs(const struct agx_device_key *dev)
{
   const unsigned halfs = dev->encoding == AGX_ISA_ENCODING_AGX3B
                             ? AGX3B_NUM_REGS
                          : dev->encoding == AGX_ISA_ENCODING_AGX3C
                             ? AGX3C_NUM_REGS
                             : AGX_NUM_REGS;

   /* The last component's 32-bit value must end inside the file, and the
    * register allocator counts one half above the last exported half
    * (agx_register_allocate.c: "We cannot use fewer registers than we
    * export" takes the exclusive end as the highest register), so that half
    * must be inside the file too. Each attribute is eight halves. */
   unsigned attribs = (halfs - 1 - AGX_ABI_VIN_ATTRIB(0)) / 8;
   assert(AGX_ABI_VIN_ATTRIB(attribs * 4 - 1) + 3 <= halfs);
   assert(AGX_ABI_VIN_ATTRIB((attribs + 1) * 4 - 1) + 3 > halfs);

   return MIN2(attribs, AGX_MAX_ATTRIBS);
}

struct agx_split_sample_tests {
   unsigned samples;
   bool m5;
};

/* Separate pure kills from testing updates before lowering their different
 * pixel-class lifetimes. The testing arm needs a real empty-EXEC skip.
 *
 * M5 keeps a whole-pixel update as one unconditional testing MOVMSK, as
 * native G17G does (agx_nir_lower_sample_mask_m5). An update naming only some
 * samples (one per sample in a sample-rate epilog) is still split.
 */
static bool
agx_nir_split_g17p_sample_tests(nir_builder *b, nir_intrinsic_instr *intr,
                                void *data)
{
   const struct agx_split_sample_tests *split = data;
   if (intr->intrinsic != nir_intrinsic_sample_mask_agx ||
       nir_src_is_const(intr->src[1]))
      return false;

   unsigned all = BITFIELD_MASK(split->samples);
   if (split->m5 && nir_src_is_const(intr->src[0]) &&
       (nir_src_as_uint(intr->src[0]) & all) == all)
      return false;
   nir_def *target = intr->src[0].ssa;
   nir_def *live = intr->src[1].ssa;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *tests = nir_iand_imm(b, nir_iand(b, target, live), all);
   nir_push_if(b, nir_ine_imm(b, tests, 0));
   nir_sample_mask_agx(b, target, live);
   nir_push_else(b, NULL);
   nir_def *active = nir_iand_imm(b, target, all);
   nir_push_if(b, nir_ieq_imm(b, active, all));
   nir_sample_mask_agx(b, nir_imm_intN_t(b, all, 16),
                       nir_imm_intN_t(b, 0, 16));
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ine_imm(b, active, 0));
   nir_sample_mask_agx(b, target, nir_imm_intN_t(b, 0, 16));
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_instr_remove(&intr->instr);
   return true;
}

/* The operands both destination-less SCS forms share: G17P only, one, two or
 * four 32-bit words at a word-aligned offset from a 32-bit base, and a value
 * nothing reads, because the access defines none. */
static bool
agx_scs_touch_nir_valid(nir_intrinsic_instr *in,
                        const struct agx_shader_key *key)
{
   const unsigned n = in->def.num_components;
   return key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          !(nir_intrinsic_flags(in) & ~AGX_NIR_SCS_TOUCH_FLAGS) &&
          in->def.bit_size == 32 && (n == 1 || n == 2 || n == 4) &&
          nir_def_is_unused(&in->def) &&
          in->src[0].ssa->bit_size == 32 && !(nir_intrinsic_base(in) & 3);
}

/* Keep unsupported private SCS forms transactional, before instruction
 * selection. These operations never reinterpret NIR scratch/global memory.
 * M5 launch setup must allocate SCS even for a load/store-only program. */
static bool
agx_validate_scs_nir(nir_shader *nir, const struct agx_shader_key *key,
                     bool *uses_scs)
{
   *uses_scs = false;
   nir_foreach_function_with_impl(func, impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            bool valid = true, memory = false, touch = false;
            switch (in->intrinsic) {
            case nir_intrinsic_load_scs_agx:
               memory = true;
               valid = in->def.num_components <= 4 &&
                       (in->def.bit_size == 32 ||
                        (in->def.bit_size == 16 && in->def.num_components == 1));
               break;
            case nir_intrinsic_store_scs_agx:
               memory = true;
               valid = in->src[0].ssa->num_components <= 4 &&
                       (in->src[0].ssa->bit_size == 32 ||
                        (in->src[0].ssa->bit_size == 16 &&
                         in->src[0].ssa->num_components == 1));
               break;
            case nir_intrinsic_load_scs_indexed_agx:
            case nir_intrinsic_store_scs_indexed_agx: {
               const bool store = in->intrinsic == nir_intrinsic_store_scs_indexed_agx;
               const nir_def *value = store ? in->src[0].ssa : &in->def;
               memory = true;
               if (!store && (nir_intrinsic_flags(in) & AGX_NIR_SCS_TOUCH)) {
                  touch = true;
                  valid = agx_scs_touch_nir_valid(in, key) &&
                          in->src[1].ssa->bit_size == 32 &&
                          (nir_intrinsic_component(in) == 2 ||
                           nir_intrinsic_component(in) == 4);
                  break;
               }
               valid = (value->bit_size == 32 || (!store && value->bit_size == 16 && value->num_components == 1)) &&
                       value->num_components <= 4 &&
                       in->src[store ? 1 : 0].ssa->bit_size == 32 &&
                       in->src[store ? 2 : 1].ssa->bit_size == 32 &&
                       nir_intrinsic_component(in) <= 4 &&
                       (store || !(nir_intrinsic_flags(in) & ~1u));
               break;
            }
            case nir_intrinsic_load_scs_register_agx: {
               unsigned format = nir_intrinsic_format(in);
               unsigned width = format == PIPE_FORMAT_R8_UINT ? 8 :
                                format == PIPE_FORMAT_R16_UINT ? 16 :
                                format == PIPE_FORMAT_R32_UINT ? 32 : 0;
               memory = true;
               valid = width && in->def.num_components <= 4 &&
                  (width == 32 || in->def.num_components == 1) &&
                  in->def.bit_size == (width == 32 ? 32 : 16) &&
                  in->src[0].ssa->bit_size == 32 &&
                  in->src[1].ssa->bit_size == 32 &&
                  !(nir_intrinsic_base(in) & (width / 8 - 1)) &&
                  nir_intrinsic_component(in) <= 4 &&
                  !(nir_intrinsic_flags(in) & ~9u) &&
                  (!(nir_intrinsic_flags(in) & 8) ||
                   (width == 8 && nir_intrinsic_component(in) == 3 &&
                    (nir_intrinsic_flags(in) & 1)));
               break;
            }
            case nir_intrinsic_load_scs_uniform_agx:
            case nir_intrinsic_store_scs_uniform_agx:
            case nir_intrinsic_load_scs_base_agx:
            case nir_intrinsic_store_scs_base_agx: {
               const bool uniform = in->intrinsic == nir_intrinsic_load_scs_uniform_agx ||
                                    in->intrinsic == nir_intrinsic_store_scs_uniform_agx;
               const bool store = in->intrinsic == nir_intrinsic_store_scs_uniform_agx ||
                                  in->intrinsic == nir_intrinsic_store_scs_base_agx;
               const nir_def *value = store ? in->src[0].ssa : &in->def;
               unsigned format = nir_intrinsic_format(in);
               unsigned width = format == PIPE_FORMAT_R8_UINT ? 8 :
                                format == PIPE_FORMAT_R16_UINT ? 16 :
                                format == PIPE_FORMAT_R32_UINT ? 32 : 0;
               unsigned flags = store ? 0 : nir_intrinsic_flags(in);
               memory = true;
               if (in->intrinsic == nir_intrinsic_load_scs_base_agx &&
                   (flags & AGX_NIR_SCS_TOUCH)) {
                  touch = true;
                  valid = agx_scs_touch_nir_valid(in, key) && width == 32 &&
                          nir_intrinsic_component(in) == 0;
                  break;
               }
               valid = width && value->num_components <= 4 &&
                  (width == 32 || value->num_components == 1) &&
                  value->bit_size == ((width == 32 || (flags & 2)) ? 32 : 16) &&
                  in->src[store ? 1 : 0].ssa->bit_size == (uniform ? 64 : 32) &&
                  !(nir_intrinsic_base(in) & (width / 8 - 1));
               if (uniform)
                  valid &= in->src[store ? 2 : 1].ssa->bit_size == 32 &&
                           nir_intrinsic_component(in) <= 4 && !(flags & ~3u) && (!(flags & 2) || width < 32);
               else
                  valid &= nir_intrinsic_component(in) <= 4 && !(flags & ~13u) &&
                           (!(flags & 4) || ((flags & 1 || key->dev.encoding == AGX_ISA_ENCODING_AGX3B) && width == 8 &&
                                            nir_intrinsic_component(in) == 4)) &&
                           (!(flags & 8) || (key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
                                            (width == 8 && nir_intrinsic_component(in) == 4)));
               break;
            }
            case nir_intrinsic_intersect_ray_gen1_agx:
               valid = key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
                  in->src[0].ssa->bit_size == 64 &&
                  in->src[1].ssa->bit_size == 64 &&
                  in->src[2].ssa->bit_size == 32 &&
                  in->src[3].ssa->bit_size == 32 &&
                  nir_intrinsic_base(in) <= 511;
               break;
            case nir_intrinsic_continue_ray_gen1_agx:
               valid = key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
                  in->src[0].ssa->bit_size == 32 && nir_intrinsic_base(in) <= 1;
               break;
            case nir_intrinsic_intersect_ray_agx:
               valid = in->src[0].ssa->bit_size == 64 &&
                       in->src[1].ssa->bit_size == 64 &&
                       in->src[2].ssa->bit_size == 32 &&
                       in->src[3].ssa->bit_size == 32 &&
                       in->src[4].ssa->bit_size == 32;
               break;
            case nir_intrinsic_enter_query_agx:
               break;
            case nir_intrinsic_continue_ray_agx:
               valid = in->src[0].ssa->bit_size == 32;
               break;
            default:
               continue;
            }
            if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B && !touch &&
                (in->intrinsic == nir_intrinsic_load_scs_uniform_agx ||
                 in->intrinsic == nir_intrinsic_load_scs_base_agx ||
                 in->intrinsic == nir_intrinsic_load_scs_indexed_agx))
               valid &= nir_intrinsic_flags(in) == 0 ||
                        (in->intrinsic == nir_intrinsic_load_scs_base_agx &&
                         (nir_intrinsic_flags(in) == 4 || nir_intrinsic_flags(in) == 8 ||
                          nir_intrinsic_flags(in) == 12) &&
                         nir_intrinsic_format(in) == PIPE_FORMAT_R8_UINT &&
                         nir_intrinsic_component(in) == 4);
            *uses_scs = true;
            if (memory)
               valid &= nir_intrinsic_base(in) >= INT16_MIN &&
                        nir_intrinsic_base(in) <= INT16_MAX;
            if (!valid || (key->dev.encoding != AGX_ISA_ENCODING_AGX3C &&
                           key->dev.encoding != AGX_ISA_ENCODING_AGX3B)) {
               agx_set_refusal(NULL, "unsupported SCS intrinsic", __LINE__);
               return false;
            }
         }
      }
   }
   return true;
}

/* Whether a fragment shader takes a derivative: ddx/ddy or a texture
 * operation with an implicit one. */
static bool
agx_fs_takes_derivatives(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex &&
                nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return true;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_ddx:
            case nir_intrinsic_ddx_fine:
            case nir_intrinsic_ddx_coarse:
            case nir_intrinsic_ddy:
            case nir_intrinsic_ddy_fine:
            case nir_intrinsic_ddy_coarse:
               return true;
            default:
               break;
            }
         }
      }
   }
   return false;
}

/* The boundary is a workgroup-wide rendezvous and shared-memory fence.
 * Divergence is an additional refusal; correctness uses the barrier's
 * workgroup execution scope on a valid shader. */
static bool
shared_phase_boundary(nir_block *block, bool entry)
{
   if (!block || block->divergent)
      return false;

   nir_instr *instr = entry ? nir_block_last_instr(block)
                            : nir_block_first_instr(block);
   if (!instr || instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *barrier = nir_instr_as_intrinsic(instr);
   return barrier->intrinsic == nir_intrinsic_barrier &&
          nir_intrinsic_execution_scope(barrier) == SCOPE_WORKGROUP &&
          nir_intrinsic_memory_scope(barrier) == SCOPE_DEVICE &&
          nir_intrinsic_memory_modes(barrier) == nir_var_mem_shared &&
          nir_intrinsic_memory_semantics(barrier) == NIR_MEMORY_ACQ_REL;
}

static bool
shared_phase_load(const nir_intrinsic_instr *intrin, UNUSED const void *data)
{
   return intrin->intrinsic == nir_intrinsic_load_shared &&
          nir_intrinsic_access(intrin) == 0 && intrin->def.bit_size == 32 &&
          intrin->def.num_components == 1;
}

/* Check the entire structured phase for writes, not just the candidate
 * block. Two workgroup rendezvous exclude stores from peers outside it. */
static bool
shared_phase_readonly(nir_loop *loop)
{
   if (!shared_phase_boundary(nir_loop_predecessor_block(loop), true) ||
       !shared_phase_boundary(nir_loop_successor_block(loop), false))
      return false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_phi:
         case nir_instr_type_load_const:
         case nir_instr_type_alu:
            break;
         case nir_instr_type_intrinsic:
            if (!shared_phase_load(nir_instr_as_intrinsic(instr), NULL))
               return false;
            break;
         case nir_instr_type_jump: {
            nir_jump_instr *jump = nir_instr_as_jump(instr);
            if (jump->type != nir_jump_break && jump->type != nir_jump_continue)
               return false;
            break;
         }
         default:
            return false;
         }
      }
   }
   return true;
}

static bool
cse_shared_reads_in_phases(nir_shader *nir)
{
   bool any = false;

   nir_divergence_analysis(nir);
   nir_foreach_function_impl(impl, nir) {
      if (impl->function->is_preamble) {
         nir_no_progress(impl);
         continue;
      }

      bool progress = false;
      nir_foreach_block(pre, impl) {
         nir_loop *loop = nir_block_get_following_loop(pre);
         if (!loop || !shared_phase_readonly(loop))
            continue;

         nir_foreach_block_in_cf_node(block, &loop->cf_node) {
            struct set loads = {0};
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               if (!loads.table)
                  nir_instr_set_init(&loads, NULL);

               /* Keep the first read in place. Do not mark it globally
                * reorderable or reuse it across a dynamic iteration. */
               if (nir_instr_set_add_or_rewrite(&loads, instr,
                                                shared_phase_load,
                                                NULL, NULL)) {
                  nir_instr_remove(instr);
                  progress = true;
               }
            }
            if (loads.table)
               nir_instr_set_fini(&loads);
         }
      }
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }

   return any;
}

bool
agx_compile_shader_nir(nir_shader *nir, struct agx_shader_key *key,
                       struct agx_shader_part *out)
{
   /* Specialization or shader linking may gather info again after
    * preprocessing, clearing the operation-derived helper flags. Execution
    * modes survive that gather and must still reach the AGX3 entry protocol
    * and the draw-time triangle-merge control, even without a surviving quad
    * intrinsic. Keep the AGX2 compilation path unchanged. */
   if (key->dev.chip == AGX_CHIP_G17P && key->dev.shader_quad_control)
      agx_preserve_quad_control(nir);

   agx_compiler_debug = agx_get_compiler_debug();
   char *refusal = agx_refusal_buffer(false);
   if (refusal)
      refusal[0] = 0;
   /* The native constant-program decision is per compile
    * (agx_shader_key::g17p_native_uniforms, read through ctx); the global
    * agx_compiler_debug is never modified here. */
   struct agx_shader_info *info = &out->info;

   struct util_dynarray binary = UTIL_DYNARRAY_INIT;

   memset(out, 0, sizeof *out);

   bool uses_scs;
   if (!agx_validate_scs_nir(nir, key, &uses_scs))
      return false;

   /* Record API/source arithmetic before backend lowering introduces its own
    * 64-bit address and invocation-ID calculations. */
   const bool source_iadd64 = agx_nir_has_alu64(nir, nir_op_iadd);
   const bool source_imul64 = agx_nir_has_alu64(nir, nir_op_imul) ||
                              agx_nir_has_alu64(nir, nir_op_umul_2x32_64) ||
                              agx_nir_has_alu64(nir, nir_op_imul_2x32_64);

   if ((key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
       has_unlowered_printf_address(nir)) {
      agx_set_refusal(NULL, "unlowered printf address", __LINE__);
      return false;
   }

   assert(nir->info.io_lowered &&
          "agx_preprocess_nir is called first, then the shader is specalized,"
          "then the specialized shader is compiled");

   /* The AGX3 backend has byte-grounded traversal/SCS codecs, but no complete
    * NIR ray-query lowering yet.  Reject before instruction selection reaches
    * its fatal unhandled-intrinsic path, and keep the output transactional.
    * This gate is removed only together with the full lowering and scheduling
    * contract; it does not prevent direct internal encoder tests. */
   if (nir->info.ray_queries != 0 &&
       (key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        key->dev.encoding == AGX_ISA_ENCODING_AGX3C)) {
      agx_set_refusal(NULL, "ray queries are not lowered on AGX3", __LINE__);
      return false;
   }

   /* If required, tag writes will be enabled by instruction selection */
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      info->tag_write_disable = !nir->info.writes_memory;

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_printf_buffer,
            nir_metadata_control_flow, &key->dev);

   if (key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
       nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_indirect_fs_input,
               nir_metadata_none, NULL);
   }

   if (agx_isa_is_agx3(key->dev.encoding) &&
       nir->info.stage == MESA_SHADER_FRAGMENT &&
       key->fs.nr_samples > 1 && !key->fs.inside_sample_loop) {
      struct agx_split_sample_tests split = {
         .samples = key->fs.nr_samples,
         .m5 = key->dev.encoding == AGX_ISA_ENCODING_AGX3C,
      };
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, agx_nir_split_g17p_sample_tests,
               nir_metadata_none, &split);
   }

   NIR_PASS(_, nir, nir_lower_frag_coord_to_pixel_coord);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* Late sysval lowering creates large loads. Load lowering creates unpacks */
   nir_lower_mem_access_bit_sizes_options lower_mem_access_options = {
      .modes = nir_var_mem_ssbo | nir_var_mem_constant |
               nir_var_mem_task_payload | nir_var_shader_temp |
               nir_var_function_temp | nir_var_mem_global | nir_var_mem_shared,
      .callback = mem_access_size_align_cb,
   };
   NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &lower_mem_access_options);

   if (agx_isa_is_agx3(key->dev.encoding)) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_g17p_halfword_tile_load,
               nir_metadata_control_flow, NULL);
   }

   /* Optimize scratch access for silly internal CL shaders */
   if (nir->info.internal) {
      NIR_PASS(_, nir, nir_lower_scratch_to_var);
      NIR_PASS(_, nir, nir_lower_vars_to_scratch, 256,
               glsl_get_natural_size_align_bytes,
               glsl_get_natural_size_align_bytes);
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_function_temp, ~0);
   }

   /* Cleanup 8-bit math before lowering */
   bool progress;
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   NIR_PASS(_, nir, nir_lower_bit_size, lower_bit_size_callback, NULL);

   /* Late blend lowering creates vectors */
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);

   NIR_PASS(_, nir, agx_nir_lower_denorm_preserve);
   if (key->dev.chip == AGX_CHIP_G17P ||
       key->dev.encoding == AGX_ISA_ENCODING_AGX3C)
      NIR_PASS(_, nir, agx_nir_lower_fma_rte);

   /* Sin/cos that agx_preprocess_nir_g17p left for an explicit-RTE shader.
    * Their reduction FMAs serve an absolute-error bound, which a tininess
    * repair of at most 2^-126 cannot affect, so they are lowered after it.
    * Every other shader has already lowered them. */
   NIR_PASS(_, nir, agx_lower_sincos, 1);

   /* Late VBO lowering creates constant udiv instructions */
   NIR_PASS(_, nir, nir_opt_idiv_const, 16);

   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_load_from_texture_handle,
            nir_metadata_control_flow, NULL);

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_sink_address,
            nir_metadata_control_flow, NULL);

   info->push_count = key->reserved_preamble;
   const bool form_preamble = !key->secondary && !key->no_preamble;
   const bool is_agx3 = key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
                       key->dev.encoding == AGX_ISA_ENCODING_AGX3C;
   const bool is_g17p = key->dev.encoding == AGX_ISA_ENCODING_AGX3B;
   const bool m5_wide_product =
      key->dev.encoding == AGX_ISA_ENCODING_AGX3C && !key->dev.emit_agx3d;

   /* Shared addresses arrive as byte offsets; AGX3 local memory instructions
    * scale scalar element indices by their access width. */
   if (is_agx3 && !nir->info.shared_element_index_agx)
      NIR_PASS(_, nir, agx_nir_lower_shared_element_index);


   /* AGX2 has native 64-bit add and 32x32->64 multiply forms, so the shared
    * compiler options preserve them.  AGX3 has no grounded 64-bit add, so
    * lower it to low/add, carry, high/add.
    *
    * G17P's 32x32->64 multiply IS grounded (agx3_iadd_core.h, dest_64), but a
    * full 64x64->64 multiply still needs several of them plus the carry
    * arithmetic, which is what agx_nir_lower_imul64 builds out of 32-bit
    * pieces.  Keep that lowering; the only cost of leaving it alone is that
    * its inner high-word multiply is expanded in software rather than using
    * one wide imad.  Changing that is a codegen optimisation and belongs with
    * a hardware measurement, not with enabling the instruction.
    *
    * Keep the override local to this compile so the other encodings retain
    * their paths and frontends continue sharing agx_nir_options. */
   const nir_shader_compiler_options *original_options = nir->options;
   nir_shader_compiler_options agx3_options;
   if (is_agx3) {
      agx3_options = *original_options;
      if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C ||
          key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* Neither encoding has a bf16 conversion. f2bf rounds to nearest
          * even here; the shared lowering, which truncates, is left only
          * bf2f. */
         NIR_PASS(_, nir, agx_nir_lower_f2bf_rtne);
         agx3_options.lower_bfloat16_conversions = true;
      }
      if (is_g17p) {
         agx3_options.lower_iabs = false;
         /* G17P's saturate form is a standalone f32add.ep2 operation. */
         agx3_options.lower_fsat = false;
         /* G17P lowers unsigned saturating subtract to compare/sub/select. */
         agx3_options.lower_usub_sat = true;
      }
      if (source_iadd64)
         agx3_options.lower_int64_options |= nir_lower_iadd64;
      if (source_imul64)
         agx3_options.lower_int64_options &= ~nir_lower_imul64;
      nir->options = &agx3_options;

      /* Lower the 64-bit multiply BEFORE agx_optimize_nir, not only after.
       *
       * Clearing nir_lower_imul64 above leaves a 64-bit imul standing all the
       * way through agx_optimize_nir, and agx_nir_fuse_algebraic_late's
       * strength-reduction rule
       *
       *    (('imul', a, 1 + (1 << s)), iaddshl(a, a, s))
       *
       * is not width-qualified, so a 64-bit `x * 3` is rewritten into a
       * 64-bit imadshl_agx.  agx_emit_imadshl_agx() then emits a 64-bit iadd,
       * which G17P has no grounded encoding for, and the whole shader is
       * refused:
       *
       *    no grounded AGX3 encoding for iadd
       *      [d64:reg4 s0=64:reg4.discard s1=64:reg4.discard]
       *
       * That refusal is reached by ordinary shaderInt64 code -- one 64-bit
       * multiply followed by a 64-bit add -- while shaderInt64 is advertised.
       * Lowering here leaves only 32-bit multiplies and adds plus
       * pack_64_2x32_split, so no rule in the late fusion can form a 64-bit
       * add out of it, and the expansion additionally gets optimized by the
       * passes that follow rather than being emitted raw.
       *
       * The existing call after agx_optimize_nir stays: this pass is
       * idempotent, and it still covers a 64-bit multiply created during
       * optimization rather than present in the source. */
      if (source_imul64)
         NIR_PASS(_, nir, agx_nir_lower_imul64, m5_wide_product);
   } else if (agx_is_g18g(&key->dev) && source_iadd64) {
      /* G18G's generated geometry shaders also add full-width addresses.
       * Apple G17G-family ulong addition uses low/add, carry and high/add;
       * keep that lowering local instead of reaching an ungrounded IADD64. */
      agx3_options = *original_options;
      agx3_options.lower_int64_options |= nir_lower_iadd64;
      nir->options = &agx3_options;
   }

   /* The AGX3 packer admits SR 34/36 only on AGX3B and in these stages
    * (agx3_pack_binary.c, GET_SR_COVERAGE); a geometry or tessellation
    * evaluation shader with a uniform-address atomic would fail to compile.
    */
   const mesa_shader_stage stage = nir->info.stage;
   const bool uniform_atomics =
      !is_agx3 ||
      (is_g17p && (stage == MESA_SHADER_COMPUTE ||
                   stage == MESA_SHADER_VERTEX ||
                   stage == MESA_SHADER_TESS_CTRL ||
                   stage == MESA_SHADER_FRAGMENT));

   /* Apple's G17P loops are a
    * guard plus a bottom-tested body (fragment and compute alike: the
    * backedge targets the first body instruction, `icmp` sits mid-body and
    * `cndlt; b` end the body). Rotate single-exit counted loops the same
    * way before the main optimization loop, so the copied test is optimized
    * with the rest. */
   if (is_agx3) {
      /* Uniform loops only (bw_copy regression, see nir_opt_loop.c). */
      nir_divergence_analysis(nir);
      bool rotated = false;
      NIR_PASS(rotated, nir, nir_opt_loop_rotate, true);
      if (rotated) {
         NIR_PASS(_, nir, nir_opt_copy_prop);
         NIR_PASS(_, nir, nir_opt_dce);
      }
   }

   /* The rotation above copies a loop's
    * header test into a guard `if (exit) {} else { loop }`, and the earlier
    * nir_opt_if runs happened before it. On AGX3 that shape costs a CNDST
    * plus a CNDEF (else flip) per invocation; inverted it is one inverted
    * CNDST, the empty else is removed by agx_opt_empty_else, and the guard
    * matches Apple's (one inverted cndst, no else). */
   if (is_agx3)
      NIR_PASS(_, nir, nir_opt_if_empty_then);

   /* Only the one-sample AGX3B coverage path reads
    * bit 0 alone; sample loops and multisampled variants keep whole masks. */
   const bool discard_word =
      ((is_g17p && key->dev.encoding == AGX_ISA_ENCODING_AGX3B) ||
       key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
      nir->info.stage == MESA_SHADER_FRAGMENT &&
      MAX2(key->fs.nr_samples, 1) == 1 && !key->fs.inside_sample_loop;

   agx_optimize_nir(
      nir, key->dev.soft_fault, uniform_atomics,
      key->dev.encoding == AGX_ISA_ENCODING_AGX3B, discard_word,
      key->dev.chip == AGX_CHIP_G17P,
      key->dev.encoding == AGX_ISA_ENCODING_AGX3C,
      key->dev.emit_agx3d, key->checked_offsets,
      form_preamble ? &info->push_count : NULL,
      (form_preamble && key->promote_textures) ? &info->texture_state_count
                                              : NULL,
      (form_preamble && key->promote_textures) ? &info->sampler_state_count
                                              : NULL,
      form_preamble ? key : NULL, info);

   /* Safe uniform-load speculation and if/phi folding have now exposed
    * matching sample copies. Preserve early subset proofs through those
    * optimizations and promote them only after complete-sample coalescing. */
   if (key->dev.emit_agx3d && nir->info.stage == MESA_SHADER_FRAGMENT &&
       (key->fs.nr_samples == 2 || key->fs.nr_samples == 4) &&
       !key->fs.inside_sample_loop)
      NIR_PASS(_, nir, agx_nir_coalesce_sample_mask_updates,
               key->fs.nr_samples);

   /* Inline blend/framebuffer-fetch shaders load the tilebuffer in the
    * main program. Final optimization can expose constant one-hot masks
    * here, after the epilog builder's sample lowering has already run. Use
    * the same coverage-aware register-mask lowering for the main program. */
   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C && key->dev.emit_agx3d &&
       nir->info.stage == MESA_SHADER_FRAGMENT &&
       (key->fs.nr_samples == 2 || key->fs.nr_samples == 4))
      NIR_PASS(_, nir, agx_nir_lower_g18_tile_samples, key->fs.nr_samples);

   if (is_agx3 && source_imul64)
      NIR_PASS(_, nir, agx_nir_lower_imul64, m5_wide_product);

   if (is_agx3)
      nir->options = original_options;

   /* AGX_G17P_UNIFORM_DMA: only when no preamble is formed. */
   if (is_agx3 && key->g17p_uniform_dma && !form_preamble &&
       agx3_nir_lower_uniform_dma(nir, key, info))
      NIR_PASS(_, nir, nir_opt_dce);

   /* Late NIR cleanup can remove main-program uses of values exported by
    * preamble extraction. Preserve their SSA uses inside the CProg, while
    * dropping writes to general uniform slots nobody reads anymore.
    */
   if (is_agx3 && form_preamble) {
      NIR_PASS(_, nir, agx_nir_remove_unused_preamble_exports,
               key->reserved_preamble);
      NIR_PASS(_, nir, agx_nir_coalesce_preamble_half_exports,
               key->reserved_preamble, info->push_count);
      NIR_PASS(_, nir, nir_opt_dce);
      /* The launch's copies, if any, sit above every export. */
      unsigned push_count = info->push_count;
      NIR_PASS(_, nir, agx_nir_place_preamble_handles_low,
               key->reserved_preamble,
               info->agx3_uniform_dma_count ? info->agx3_uniform_dma[0].dest
                                            : push_count,
               &push_count, is_g17p);
      info->push_count = push_count;
   }

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      info->varyings.fs.nr_cf = key->fs.cf_base;
      assign_coefficient_regs(nir, &info->varyings.fs);
   }

   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
      /* Hoist only what the register file can hold through the loop: the
       * file a workgroup allows (agx_ra), less what spilling reserves.
       */
      unsigned regs = ROUND_DOWN_TO(AGX3C_NUM_REGS, 8);
      if (mesa_shader_stage_is_compute(nir->info.stage)) {
         const unsigned threads = nir->info.workgroup_size_variable
            ? 1024
            : nir->info.workgroup_size[0] * nir->info.workgroup_size[1] *
                 nir->info.workgroup_size[2];
         regs = MIN2(regs, agx_max_registers_for_occupancy(threads));
      }

      NIR_PASS(_, nir, agx_nir_opt_licm, regs - 8);
   }

   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
      NIR_PASS(_, nir, agx_nir_opt_shared_byte_offsets);
      NIR_PASS(_, nir, nir_opt_cse);
      NIR_PASS(_, nir, nir_opt_dce);
   }

   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
      NIR_PASS(_, nir, agx_nir_opt_canonical_minmax);
      NIR_PASS(_, nir, nir_opt_dce);
   }


   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
      bool guarded = false;
      NIR_PASS(guarded, nir, nir_shader_intrinsics_pass,
               m5_lower_predicated_load, nir_metadata_none, NULL);
      if (guarded) {
         /* Address/ALU/phi lowering and dead-code cleanup do not move loads
          * out of these arms. No speculation, LICM or if-flattening follows. */
         NIR_PASS(_, nir, agx_nir_lower_address, true, true,
                  key->checked_offsets);
         NIR_PASS(_, nir, nir_lower_int64);
         NIR_PASS(_, nir, nir_opt_constant_folding);
         NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
         NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
         NIR_PASS(_, nir, nir_lower_all_phis_to_scalar);
         NIR_PASS(_, nir, nir_opt_copy_prop);
         NIR_PASS(_, nir, nir_opt_dce);
         nir_divergence_analysis(nir);
      }
   }

   /* Apple issues a loop body's samples in source order and consumes all of
    * them after the last one (native robust_loop witness); Mesa consumed each
    * right after its sample. Geometry-side M5 stages, like the M5 copy-loop
    * schedule; see agx_nir_opt_sink_loop_sample_uses.c. */
   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
       nir->info.stage != MESA_SHADER_FRAGMENT &&
       nir->info.stage != MESA_SHADER_COMPUTE) {
      NIR_PASS(_, nir, agx_nir_opt_sink_loop_sample_uses);

      /* Every control-flow instruction retires the samples in flight, so
       * also issue the body's unconditional samples ahead of the
       * side-effect-free conditional blocks before them, then merge the
       * conditional blocks this leaves adjacent; see
       * agx_nir_opt_hoist_loop_samples. */
      bool hoisted = false;
      NIR_PASS(hoisted, nir, agx_nir_opt_hoist_loop_samples);
      bool merged = false;
      if (hoisted)
         NIR_PASS(merged, nir, nir_opt_if_merge);
      if (merged) {
         NIR_PASS(_, nir, nir_opt_cse);
         NIR_PASS(_, nir, nir_opt_dce);
      }
   }

   if (nir->info.stage == MESA_SHADER_COMPUTE && !nir->info.internal &&
       !key->secondary) {
      /* Generic NIR: a read-only phase between two workgroup barriers sees
       * no shared store, so this holds for every chip. */
      if (nir->info.shared_size)
         NIR_PASS(_, nir, cse_shared_reads_in_phases);

      /* Every AGX3B part returns a four-lane W8 load as one byte-packed
       * word (the W8 path of agx_emit_load()), which the fold extracts with
       * integer shifts. AGX2 returns one byte per 16-bit lane, so there is no
       * word to fold to. AGX3C (M5, and G18 through it) widens bytes to
       * 32-bit lanes with its own unpack in agx_emit_load(), so this AGX3B
       * extraction is not its form. */
      if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, fold_w8_word_unpack,
                  nir_metadata_control_flow, NULL);
   }

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      nir_print_shader(nir, stdout);

   if (is_g17p) {
      info->agx3_uniform_halfs_valid = true;
      nir_foreach_function_with_impl(func, impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_load_preamble)
                  continue;
               unsigned base = nir_intrinsic_base(intr);
               unsigned halves = intr->def.num_components *
                                 DIV_ROUND_UP(intr->def.bit_size, 16);
               if (base >= AGX3_CPROG_INPUT_HALVES ||
                   halves > AGX3_CPROG_INPUT_HALVES - base) {
                  info->agx3_uniform_halfs_valid = false;
                  continue;
               }
               for (unsigned h = base; h < base + halves; ++h)
                  info->agx3_uniform_halfs[h / 64] |= BITFIELD64_BIT(h % 64);
            }
         }
      }
   }

   info->local_size = nir->info.shared_size;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_cmat_muladd_agx &&
                   intr->def.num_components == 8)
                  info->uses_matrix16 = true;
            }
         }
      }
   }
   info->uses_resource_info_query = nir->info.uses_resource_info_query;
   info->uses_ray_traversal = nir->info.ray_queries != 0 || uses_scs;
   /* NIR ray_queries counts inline intersection_query variables, matching the
    * G17S next()/committed witness whose ShaderInfo ID32 is nonzero.  Future
    * one-shot/full traversal lowerings must set uses_ray_traversal without
    * automatically setting this committed-path property. */
   info->ray_traversal_depth_nonzero = nir->info.ray_queries != 0;

   nir_foreach_function_with_impl(func, impl, nir) {
      unsigned offset;
      const struct agx_shader_info saved_info = out->info;
      const struct agx_shader_key saved_key = *key;
      const unsigned saved_size = binary.size;
      bool compiled = agx_compile_function_nir(nir, impl, key, &binary,
                                               &out->info, &offset, false);

      /* The G17G packer refuses shapes it has no grounded encoding for and
       * rolls the binary back. Most refusals in real clients (Skia/ANGLE)
       * are float constants inlined as minifloat immediates; the same
       * function with those constants in registers encodes. */
      if (!compiled && key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
         if (agx_compiler_debug & AGX_DBG_SHADERS)
            fprintf(stderr, "AGX3: retrying with float constants in registers\n");
         out->info = saved_info;
         *key = saved_key;
         binary.size = saved_size;
         compiled = agx_compile_function_nir(nir, impl, key, &binary,
                                             &out->info, &offset, true);
      }

      if (!compiled) {
         util_dynarray_fini(&binary);
         memset(out, 0, sizeof(*out));
         return false;
      }

      if (func->is_preamble) {
         info->preamble_offset = offset;
         info->preamble_size = binary.size - offset;
         info->has_preamble = true;
      } else if (func->is_entrypoint) {
         info->main_offset = offset;
         /* The relocation list follows the main program. */
         info->main_size = (info->agx3_uvs_reloc_count
                               ? info->agx3_uvs_relocs_offset
                               : binary.size) - offset;
      } else {
         UNREACHABLE("General functions not yet supported");
      }
   }

   /* The preamble may be compiled after main and legalize a final half-word
    * export into a native dword. Report the complete physical footprint. */
   info->stats.uniforms = info->push_count;

   if (agx_should_dump(nir, AGX_DBG_SHADERDB)) {
      agx2_stats_fprintf(stderr, _mesa_shader_stage_to_abbrev(nir->info.stage),
                         &out->info.stats);
      if (agx_isa_is_agx3(key->dev.encoding) && info->has_preamble) {
         fprintf(stderr, "AGX3 CProg: inputs=%u storage=%u scratch=%u "
                         "gpr_halves=%u bytes=%u native_tps=%u "
                         "staging_halfs=%u staging_known=%u\n",
                 info->agx3_preamble_inputs.count,
                 info->agx3_preamble_inputs.storage_bytes,
                 info->preamble_scratch_size, info->nr_preamble_gprs,
                 info->preamble_size, info->agx3_preamble_tps_inputs,
                 info->agx3_preamble_staging_halfs,
                 info->agx3_preamble_staging_known);
      }
   }

   info->stage = nir->info.stage;

   /* Check these outside the stage check since nir->info.stage is the hardware
    * stage and these are read in the vertex *software* stage.
    */
   info->uses_draw_id =
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_DRAW_ID);

   info->uses_base_param =
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_FIRST_VERTEX) ||
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_BASE_VERTEX) ||
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_BASE_INSTANCE);

   if (nir->info.stage == MESA_SHADER_VERTEX ||
       nir->info.stage == MESA_SHADER_TESS_EVAL) {
      info->nonzero_viewport = nir->info.outputs_written & VARYING_BIT_VIEWPORT;

      info->writes_layer_viewport =
         nir->info.outputs_written & (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT);

   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      info->disable_tri_merging =
         nir->info.uses_wide_subgroup_intrinsics ||
         nir->info.fs.needs_coarse_quad_helper_invocations ||
         nir->info.writes_memory;

      /* Quad intrinsics alone do not count: they read another lane's own
       * value, which stays correct when a merged quad holds two primitives,
       * and a merged quad is still full. spirv_to_nir sets require_full_quads
       * for every quad operation, so only QuadDerivativesKHR counts here. */
      info->needs_quad_derivatives =
         (nir->info.fs.needs_coarse_quad_helper_invocations ||
          nir->info.fs.needs_full_quad_helper_invocations) &&
         (nir->info.fs.quad_derivatives || agx_fs_takes_derivatives(nir));

      info->needs_full_quads =
         nir->info.fs.needs_full_quad_helper_invocations;

      /* Writing the sample mask requires tag writes */
      info->tag_write_disable &= !info->writes_sample_mask;

      /* Report a canonical depth layout. This happens at the end because the
       * sample mask lowering affects it.
       */
      enum gl_frag_depth_layout layout = nir->info.fs.depth_layout;

      if (!(nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)))
         info->depth_layout = FRAG_DEPTH_LAYOUT_UNCHANGED;
      else if (layout == FRAG_DEPTH_LAYOUT_NONE)
         info->depth_layout = FRAG_DEPTH_LAYOUT_ANY;
      else
         info->depth_layout = layout;

      info->reads_tib = nir->info.fs.uses_fbfetch_output;
      info->early_fragment_tests = nir->info.fs.early_fragment_tests;
   } else if (nir->info.stage == MESA_SHADER_COMPUTE ||
              nir->info.stage == MESA_SHADER_MESH ||
              nir->info.stage == MESA_SHADER_TASK) {
      if (nir->info.stage == MESA_SHADER_COMPUTE)
         info->imageblock_stride = nir->info.cs.image_block_size_per_thread_agx;

      for (unsigned i = 0; i < 3; ++i) {
         info->workgroup_size[i] = nir->info.workgroup_size[i];
      }
   }

   out->binary = binary.data;
   info->binary_size = binary.size;
   return true;
}

void
agx_preprocess_nir_m5(nir_shader *nir)
{
   agx_preprocess_nir_internal(nir, false, true, false, true, false);
}
