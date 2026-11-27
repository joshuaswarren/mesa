/*
 * Copyright 2023 Valve Corporation
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * Copyright 2016 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compiler.h"
#include "agx_nir.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_abi.h"
#include "agx_nir_texture.h"
#include "glsl_types.h"
#include "libagx.h"
#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"
#include "util/format/u_format.h"

struct agx_texture_lower_options {
   bool gen9;
   bool gen10;
   /* G18G (AGX_NIR_TEXTURE_GENERATION_G18G): Gen10 plus the forms the J873
    * GL 4.6 campaign qualified. */
   bool g18g;
   bool buffer_metadata;
   /* AGX_NIR_TEXTURE_IMAGE_SIDEBAND: Gen10 image atomics read their
    * addressing word after the texture/PBE pair, not from the PBE. */
   bool image_sideband;
   bool level_zero_extent;
   agx_buffer_texture_metadata_cb buffer_metadata_cb;
   /* Formatless image loads still need view-bounds robustness. */
   bool robust_formatless_images;
   /* G17P texel fetches need explicit view-bounds checks (image robustness). */
   bool robust_texel_fetch;
   /* AGX_NIR_TEXTURE_HW_TXF_BOUNDS: hardware enforces view bounds and
    * missing-alpha one on integer texelFetch (agx_chip_hw_texel_fetch_bounds);
    * saturate wide layer, LOD and sample indices before 16-bit narrowing and
    * validate nonzero LODs and multisample indices against the view. */
   bool hw_txf_bounds;
   /* Offset between a sparse resource's read/write mapping and its read-only
    * alias; see agx_device::sparse_ro_offset.  The residency map is indexed
    * from the primary mapping, so a descriptor that addresses the alias has
    * to have it taken back off.  Masking is a no-op for a descriptor that
    * already names the primary mapping, which is what hk_buffer_view.c emits
    * on G17P, so the same code covers both. */
   uint64_t sparse_ro_offset;

   /* Mask an ordinary (non-sparse-opcode) texel-buffer read of a sparse
    * residency buffer to zero when its page is unbound.  The descriptor names
    * the primary mapping, so nothing else delivers
    * sparseResidencyNonResidentStrict for these reads. */
   bool sparse_buffer_residency;
};

static nir_def *
load_buffer_metadata(nir_builder *b, nir_def *descriptor, unsigned offset,
                     unsigned bits)
{
   return nir_load_global_constant(b, 1, bits,
      nir_iadd_imm(b, descriptor, offset), .align_mul = bits / 8);
}

/* Residency flags are inverted from NIR */
#define AGX_RESIDENT (0)

static bool
fence_image(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_after_instr(&intr->instr);

   /* If the image is write-only, there is no fencing needed */
   if (nir_intrinsic_has_access(intr) &&
       (nir_intrinsic_access(intr) & ACCESS_NON_READABLE)) {
      return false;
   }

   switch (intr->intrinsic) {
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_store:
      nir_fence_pbe_to_tex_agx(b);
      return true;

   case nir_intrinsic_image_atomic:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_bindless_image_atomic_swap:
      nir_fence_mem_to_tex_agx(b);
      return true;

   default:
      return false;
   }
}

static nir_def *
texture_descriptor_ptr(nir_builder *b, nir_tex_instr *tex)
{
   int handle_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   assert(handle_idx >= 0 && "must be bindless");
   return nir_load_from_texture_handle_agx(b, tex->src[handle_idx].src.ssa);
}

static nir_def *
buffer_texture_descriptor_ptr(nir_builder *b, nir_tex_instr *tex,
                                const struct agx_texture_lower_options *options)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF && options &&
       options->buffer_metadata_cb)
      return options->buffer_metadata_cb(b, tex);
   return texture_descriptor_ptr(b, tex);
}

static bool
bound_image_atomic(const nir_intrinsic_instr *intr, const void *data)
{
   return (intr->intrinsic == nir_intrinsic_image_atomic ||
           intr->intrinsic == nir_intrinsic_image_atomic_swap) &&
          intr->def.bit_size == 32;
}

static bool
fence_bound_image_atomic(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   return bound_image_atomic(intr, NULL) && fence_image(b, intr, NULL);
}

/* Gallium G18 supplies explicit view metadata at binding lowering. Preserve
 * the same fences as the later image lowering before removing the atomics. */
bool
agx_nir_lower_bound_image_atomics(nir_shader *s)
{
   nir_shader_intrinsics_pass(s, fence_bound_image_atomic,
                              nir_metadata_control_flow, NULL);
   return nir_lower_image_atomics_to_global(s, bound_image_atomic, NULL);
}

static bool
bound_ms_image_store(const nir_intrinsic_instr *intr, const void *data)
{
   if (intr->intrinsic != nir_intrinsic_image_store ||
       nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_MS ||
       intr->src[3].ssa->bit_size != 32)
      return false;

   switch (nir_intrinsic_format(intr)) {
   case PIPE_FORMAT_NONE:
      /* Formatless writes use the actual bound view's raw32 block size.
       * The Gallium metadata validator rejects packed/normalized formats
       * until their conversion lowering is implemented. */
      return true;
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
      return true;
   default:
      return false;
   }
}

static bool
lower_bound_ms_image_store(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (!bound_ms_image_store(intr, NULL))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   enum pipe_format format = nir_intrinsic_format(intr);
   unsigned components = util_format_get_blocksize(format) / 4;
   nir_def *address = nir_image_texel_address(
      b, 64, intr->src[0].ssa, intr->src[1].ssa, intr->src[2].ssa,
      .image_dim = GLSL_SAMPLER_DIM_MS,
      .image_array = nir_intrinsic_image_array(intr), .format = format,
      .access = nir_intrinsic_access(intr));
   nir_intrinsic_set_range_base(nir_def_as_intrinsic(address),
                                nir_intrinsic_range_base(intr));
   if (format == PIPE_FORMAT_NONE) {
      nir_def *bytes = nir_load_image_block_size_agx(b, intr->src[0].ssa);
      /* R/RG/RGBA raw32 views occupy 4/8/16 bytes. Never write unused
       * components into the adjacent pixel or sample. The view may change
       * between dispatches without recompiling this formatless program. */
      for (unsigned c = 0; c < 4; ++c) {
         nir_push_if(b, nir_uge_imm(b, bytes, (c + 1) * 4));
         nir_store_global(b, nir_channel(b, intr->src[3].ssa, c),
                          nir_iadd_imm(b, address, c * 4), .align_mul = 4,
                          .write_mask = 1, .access = nir_intrinsic_access(intr));
         nir_pop_if(b, NULL);
      }
   } else {
      nir_store_global(b, nir_trim_vector(b, intr->src[3].ssa, components), address,
                       .align_mul = 4, .write_mask = BITFIELD_MASK(components),
                       .access = nir_intrinsic_access(intr));
   }
   if (!(nir_intrinsic_access(intr) & ACCESS_NON_READABLE))
      nir_fence_mem_to_tex_agx(b);

   b->shader->info.use_lowered_image_to_global = true;
   nir_instr_remove(&intr->instr);
   return true;
}

bool
agx_nir_lower_bound_ms_image_stores(nir_shader *s)
{
   /* Raw memory lacks the image hardware's bounds checks. Guard coordinates
    * and the sample index before removing the image operation, regardless
    * of whether the GL context requested robust buffer access. */
   bool progress = nir_lower_robust_access(s, bound_ms_image_store, NULL);
   return nir_shader_intrinsics_pass(s, lower_bound_ms_image_store,
                                     nir_metadata_none, NULL) || progress;
}

static bool
has_nonzero_lod(nir_tex_instr *tex)
{
   int idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (idx < 0)
      return false;

   nir_src src = tex->src[idx].src;
   return !(nir_src_is_const(src) && nir_src_as_uint(src) == 0);
}

/*
 * The robustImageAccess2 value of an invalid read on G17P: zero, with a one
 * alpha when the bound view's format has none, which its Gen9 texture
 * descriptor records as a constant-one alpha swizzle.
 */
static nir_def *
g17p_robust_oob_value(nir_builder *b, nir_def *texture_descriptor,
                      nir_alu_type dest_type, unsigned num_components,
                      unsigned bit_size)
{
   nir_def *zero = nir_imm_zero(b, num_components, bit_size);
   if (num_components != 4)
      return zero;

   nir_def *one = nir_alu_type_get_base_type(dest_type) == nir_type_float
                     ? nir_imm_floatN_t(b, 1.0, bit_size)
                     : nir_imm_intN_t(b, 1, bit_size);
   nir_def *alpha_one = libagx_image_oob_alpha_one_gen9(b, texture_descriptor);
   return nir_vector_insert_imm(
      b, zero, nir_bcsel(b, alpha_one, one, nir_channel(b, zero, 3)), 3);
}

static bool
lower_tex_crawl(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   b->cursor = nir_before_instr(instr);

   if (tex->op != nir_texop_txs && tex->op != nir_texop_texture_samples &&
       tex->op != nir_texop_query_levels)
      return false;

   const struct agx_texture_lower_options *options = data;
   nir_def *ptr = buffer_texture_descriptor_ptr(b, tex, options);
   unsigned nr_comps = tex->def.num_components;
   assert(nr_comps <= 3);

   int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   nir_def *lod = lod_idx >= 0 ? nir_u2u16(b, tex->src[lod_idx].src.ssa)
                               : nir_imm_intN_t(b, 0, 16);

   nir_def *res;
   const bool gen9 = options && options->gen9;
   if (options && options->buffer_metadata && tex->op == nir_texop_txs &&
       tex->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
      res = load_buffer_metadata(b, ptr, AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET, 32);
   } else if (options && options->gen10 && tex->op == nir_texop_txs &&
              tex->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
      /* Gallium's Gen10 buffer texture keeps its element count in the
       * descriptor's last word (agx_pack_texture), where Gen10 leaves it
       * free; Gen10 has no G17P metadata record. */
      res = load_buffer_metadata(b, ptr, AGX_GEN10_BUFFER_TEXTURE_SIZE_OFFSET, 32);
   } else if (options && options->level_zero_extent && tex->op == nir_texop_txs &&
              tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_array &&
              nr_comps == 2 && nir_scalar_is_zero(nir_get_scalar(lod, 0))) {
      /* The packed word is zero for a null image. A Vulkan 2D view always
       * has two nonzero dimensions, each at most 32768. */
      nir_def *extent = load_buffer_metadata(b, ptr, AGX_GEN10_IMAGE_EXTENT_OFFSET, 32);
      res = nir_vec2(b, nir_iand_imm(b, extent, 0xffff),
                     nir_ushr_imm(b, extent, 16));
   } else if (gen9 && tex->op == nir_texop_txs &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_BUF) {
      res = libagx_txs_gen9(b, ptr, lod, nir_imm_int(b, nr_comps),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_1D),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_2D),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE),
                    nir_imm_bool(b, tex->is_array));
   } else if (gen9 && tex->op == nir_texop_query_levels) {
      res = libagx_texture_levels_gen9(b, ptr);
   } else if (gen9 && tex->op == nir_texop_texture_samples) {
      res = libagx_texture_samples_gen9(b, ptr);
   } else if (options && options->gen10 && tex->op == nir_texop_txs &&
              /* G18G sizes a buffer through the generic crawl below. */
              !(options->g18g && tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)) {
      res = libagx_txs_gen10(b, ptr, lod, nir_imm_int(b, nr_comps),
                           nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_1D),
                           nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_2D),
                           nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE),
                           nir_imm_bool(b, tex->is_array));
   } else if (options && options->gen10 && tex->op == nir_texop_query_levels) {
      res = libagx_texture_levels_gen10(b, ptr);
   } else if (options && options->gen10 && tex->op == nir_texop_texture_samples) {
      res = libagx_texture_samples_gen10(b, ptr);
   } else if (tex->op == nir_texop_txs) {
      res =
         libagx_txs(b, ptr, lod, nir_imm_int(b, nr_comps),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_BUF),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_1D),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_2D),
                    nir_imm_bool(b, tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE),
                    nir_imm_bool(b, tex->is_array));
   } else if (tex->op == nir_texop_query_levels) {
      res = libagx_texture_levels(b, ptr);
   } else {
      res = libagx_texture_samples(b, ptr);
   }

   nir_def_rewrite_uses(&tex->def, nir_trim_vector(b, res, nr_comps));
   nir_instr_remove(instr);
   return true;
}

/*
 * Is the page backing one texel of a sparse texel buffer bound?
 *
 * `element` is the access's coordinate in the view, which counts from the
 * view's exact address (AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET) without the
 * texture's tail offset.  The descriptor carries the
 * residency map's page number and the view's element size, and the map heads
 * with the resource's base address and its page count, so the lookup is three
 * constant loads and a bit test.
 *
 * Returns true for a descriptor with no residency map -- an ordinary buffer,
 * or a format whose element size is not a power of two -- so callers do not
 * have to special-case one.  Out-of-bounds is the caller's business: hardware
 * robustness already returns a defined zero texel for it.
 */
static nir_def *
buffer_texel_resident(nir_builder *b, nir_def *desc, nir_def *element,
                      const struct agx_texture_lower_options *options)
{
   nir_def *sparse = load_buffer_metadata(
      b, desc, AGX_G17P_BUFFER_TEXTURE_SPARSE_OFFSET, 32);
   nir_def *map_page =
      nir_ushr_imm(b, sparse, AGX_G17P_BUFFER_TEXTURE_SPARSE_MAP_SHIFT);
   nir_def *elsize_log2 =
      nir_iand_imm(b, sparse, AGX_G17P_BUFFER_TEXTURE_SPARSE_ELSIZE_MASK);
   nir_def *has_map = nir_ine_imm(b, map_page, 0);

   /* Every global load below is speculative: without a map they address the
    * zero page rather than whatever the packed word happens to hold. */
   static_assert(AIL_PAGESIZE == (1u << 14), "update the page shift");
   nir_def *map_base = nir_ishl_imm(b, nir_u2u64(b, map_page), 14);
   nir_def *safe_map_base = nir_bcsel(b, has_map, map_base,
                                      nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
   nir_def *resource_base = nir_load_global_constant(
      b, 1, 64,
      nir_iadd_imm(b, safe_map_base, AGX_SPARSE_BUFFER_MAP_RESOURCE_BASE_OFFSET),
      .align_mul = 8, .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   nir_def *page_count = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, safe_map_base, AGX_SPARSE_BUFFER_MAP_PAGE_COUNT_OFFSET),
      .align_mul = 4, .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);

   nir_def *view_base = nir_iand_imm(
      b, load_buffer_metadata(b, desc, AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET,
                              64),
      ~options->sparse_ro_offset);
   nir_def *access =
      nir_iadd(b, view_base, nir_ishl(b, nir_u2u64(b, element), elsize_log2));
   nir_def *page = nir_ushr_imm(b, nir_isub(b, access, resource_base), 14);
   nir_def *in_range = nir_ult(b, page, nir_u2u64(b, page_count));

   nir_def *word_addr = nir_iadd_imm(
      b, nir_iadd(b, map_base, nir_ishl_imm(b, nir_ushr_imm(b, page, 5), 2)),
      AGX_SPARSE_BUFFER_MAP_BITMAP_OFFSET);
   nir_def *safe_word_addr =
      nir_bcsel(b, nir_iand(b, has_map, in_range), word_addr,
                nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
   nir_def *word = nir_load_global_constant(
      b, 1, 32, safe_word_addr, .align_mul = 4,
      .access = ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   /* Test the bit with a variable MASK, not `(word >> (page & 31)) & 1`.
    *
    * That shape is folded by agx_nir_fuse_algebraic_late into
    * ubitfield_extract(word, page, 1) -- nir_opt_algebraic has already
    * dropped the `& 31`, because NIR defines a shift to mask its count --
    * and that fuse pass runs BEFORE agx_nir_lower_algebraic_late, whose
    * lower_sm5_shift is what puts the mask back for AGX, and which covers
    * only ishl/ishr/ushr, never the fused extract.  So the raw page number
    * reaches the hardware as the extract's offset.
    *
    * Confirmed in the compiled NIR (`ubitfield_extract %86, %73, 1` with %73
    * the unmasked page) and on Neo (G17P, A18 Pro): masking the count to
    * 0..15, which NIR cannot remove, passes
    * uniform_texel_buffer_sparse_fetch_size_2_24_r32ui, and masking to 0..31,
    * which it does remove, fails it -- the 16 MB buffer spans 1024 pages, so
    * every page past 31 read back non-resident.  A variable mask keeps the
    * count on an ishl, which lower_sm5_shift does mask.
    */
   nir_def *bit = nir_iand(
      b, word,
      nir_ishl(b, nir_imm_int(b, 1), nir_iand_imm(b, nir_u2u32(b, page), 31)));

   return nir_ior(b, nir_inot(b, has_map),
                  nir_iand(b, in_range, nir_ine_imm(b, bit, 0)));
}

/*
 * Sparse residency for a texel buffer.
 *
 * A sparse texel fetch on an IMAGE gets its residency code from the hardware,
 * because hk_image_view.c puts the texture descriptor in AGX_IMAGE_MODE_SPARSE
 * and points it at a page table.  A texel BUFFER never gets that treatment:
 * hk_buffer_view.c builds a plain linear 2D texture over the buffer's
 * read-only alias, so the texture unit has nothing to report residency from
 * and answers "resident" for every page, bound or not.  Measured on Neo (G17P,
 * A18 Pro): with the hardware sparse fetch emitted,
 * dEQP-VK.sparse_resources.buffer.texel_buffers.sparse_residency.
 * uniform_texel_buffer_sparse_fetch_size_2_{10,16,24}_r32ui{,_strict} passed
 * only the two size_2_10 cases, whose buffer is a single page and therefore
 * fully resident; every case with an unbound page failed.
 *
 * So take the residency from the same per-page bitmap the sparse SSBO path
 * reads (lower_sparse_buffer_deref() in hk_nir_lower_descriptors.c) and drop
 * the hardware residency request.  That also keeps this shape away from the
 * AGX3 sparse fetch encoder, which refuses it.
 */
static bool
lower_sparse_buffer_texture(nir_builder *b, nir_instr *instr, void *data)
{
   const struct agx_texture_lower_options *options = data;

   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (!tex->is_sparse || tex->sampler_dim != GLSL_SAMPLER_DIM_BUF)
      return false;

   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord_idx < 0)
      return false;

   b->cursor = nir_before_instr(instr);

   nir_def *desc = buffer_texture_descriptor_ptr(b, tex, options);
   nir_def *coord = tex->src[coord_idx].src.ssa;

   nir_def *elements =
      load_buffer_metadata(b, desc, AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET, 32);
   nir_def *oob = nir_uge(b, coord, elements);

   /* An out-of-bounds fetch returns a defined zero texel through hardware
    * robustness, so call it resident rather than inventing a page for it. */
   nir_def *resident =
      nir_ior(b, oob, buffer_texel_resident(b, desc, coord, options));

   /* AGX_RESIDENT is the resident code; agx_nir_lower_texture's
    * is_sparse_texels_resident lowering tests bit 0. */
   nir_def *code = nir_bcsel(b, resident, nir_imm_int(b, AGX_RESIDENT),
                             nir_imm_int(b, AGX_RESIDENT ^ 1));

   /* Drop the hardware residency request and re-attach ours. */
   const unsigned dense = tex->def.num_components - 1;
   tex->is_sparse = false;
   tex->def.num_components = dense;

   b->cursor = nir_after_instr(instr);

   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < dense; ++i)
      comps[i] = nir_channel(b, &tex->def, i);
   comps[dense] = nir_u2uN(b, code, tex->def.bit_size);

   nir_def *full = nir_vec(b, comps, dense + 1);
   nir_def_rewrite_uses_after(&tex->def, full);
   return true;
}

/*
 * Given a 1D buffer texture coordinate, calculate the 2D coordinate vector that
 * will be used to access the linear 2D texture bound to the buffer.
 */
static nir_def *
coords_for_buffer_texture(nir_builder *b, nir_def *coord)
{
   return nir_vec2(b, nir_umod_imm(b, coord, AGX_TEXTURE_BUFFER_WIDTH),
                   nir_udiv_imm(b, coord, AGX_TEXTURE_BUFFER_WIDTH));
}

/*
 * Buffer textures are lowered to 2D (1024xN) textures in the driver to access
 * more storage. When lowering, we need to fix up the coordinate accordingly.
 *
 * Furthermore, RGB32 formats are emulated by lowering to global memory access,
 * so to read a buffer texture we generate code that looks like:
 *
 *    if (descriptor->format == RGB32)
 *       return ((uint32_t *) descriptor->address)[x];
 *    else
 *       return txf(texture_as_2d, vec2(x % 1024, x / 1024));
 */
static bool
lower_buffer_texture(nir_builder *b, nir_tex_instr *tex, bool buffer_metadata,
                     const struct agx_texture_lower_options *options,
                     int fixed_format, bool known_in_bounds, nir_def **result)
{
   nir_def *coord = nir_steal_tex_src(tex, nir_tex_src_coord);
   nir_def *bound_metadata = NULL;
   if (tex->backend_flags & AGX_TEXTURE_FLAG_BUFFER_METADATA) {
      bound_metadata = nir_steal_tex_src(tex, nir_tex_src_backend2);
      assert(bound_metadata && bound_metadata->num_components == 4 &&
             bound_metadata->bit_size == 32);
      tex->backend_flags &= ~AGX_TEXTURE_FLAG_BUFFER_METADATA;
   }
   nir_def *size = bound_metadata ? nir_channel(b, bound_metadata, 0)
                                  : nir_get_texture_size(b, tex);
   nir_def *oob = known_in_bounds ? nir_imm_bool(b, false) :
      nir_uge(b, coord, size);

   /* Apply the buffer offset after calculating oob but before remapping */
   nir_def *desc =
      bound_metadata ? NULL : buffer_texture_descriptor_ptr(b, tex, options);
   nir_def *view_coord = coord;
   coord = bound_metadata ? coord : buffer_metadata
      ? nir_iadd(b, coord, load_buffer_metadata(
           b, desc, AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET, 32))
      : libagx_buffer_texture_offset(b, desc, coord);

   /* The descriptor names the primary mapping, so an unbound page reads the
    * shared scratch page rather than zero.  Decide residency from the texel's
    * own index, before the out-of-bounds remap below destroys it, and mask the
    * result at the end.  A sparse fetch has already had its residency code
    * attached by lower_sparse_buffer_texture(), which clears is_sparse; its
    * data still has to be masked here, and CSE folds the two lookups. */
   nir_def *resident =
      (buffer_metadata && options && options->sparse_buffer_residency)
         ? buffer_texel_resident(b, desc, view_coord, options)
         : NULL;

   /* Map out-of-bounds indices to out-of-bounds coordinates for robustness2
    * semantics from the hardware.
    */
   coord = nir_bcsel(b, oob, nir_imm_int(b, -1), coord);

   bool is_float = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;

   /* Lower RGB32 reads if the format requires. If we are out-of-bounds, we use
    * the hardware path so we get a zero texel.
    */
   nir_def *is_rgb32 = fixed_format >= 0
      ? nir_imm_bool(b, fixed_format)
      : bound_metadata
      ? nir_ine_imm(b, nir_channel(b, bound_metadata, 1), 0)
      : buffer_metadata
      ? nir_ine_imm(b, load_buffer_metadata(
           b, desc, AGX_G17P_BUFFER_TEXTURE_RGB32_OFFSET, 32), 0)
      : libagx_texture_is_rgb32(b, desc);
   nir_if *nif = nir_push_if(b, nir_iand(b, is_rgb32, nir_inot(b, oob)));

   nir_def *rgb32;
   if (bound_metadata || buffer_metadata) {
      nir_def *base = bound_metadata
         ? nir_pack_64_2x32(b, nir_channels(b, bound_metadata, 0xc))
         : load_buffer_metadata(
              b, desc, AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET, 64);
      /* In-bounds RGB32 buffer indices fit the 32-bit multiply by three,
       * as in libagx_texture_load_rgb32. Keep the address add fully 64-bit.
       */
      nir_def *address = nir_iadd(b, base,
         nir_ishl_imm(b, nir_u2u64(b, nir_imul_imm(b, view_coord, 3)), 2));
      nir_def *rgb = nir_load_global_constant(b, 3, 32, address, .align_mul = 4);
      rgb32 = nir_vec4(b, nir_channel(b, rgb, 0), nir_channel(b, rgb, 1),
                       nir_channel(b, rgb, 2),
                       is_float ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1));
   } else {
      rgb32 = libagx_texture_load_rgb32(b, desc, view_coord,
                                        nir_imm_bool(b, is_float));
   }
   rgb32 = nir_trim_vector(b, rgb32, nir_tex_instr_result_size(tex));

   /* Raw loads do not return residency information, but residency queries are
    * supported on buffer textures. Fortunately, we do not need to support
    * sparse RGB32 buffers, so we simply claim all RGB32 loads were resident.
    * Nothing should hit this in practice, but if we don't do *something* here
    * we'll get vector size mismatches which blow up in vkd3d-proton.
    */
   if (tex->is_sparse) {
      rgb32 = nir_pad_vector_imm_int(b, rgb32, AGX_RESIDENT,
                                     rgb32->num_components + 1);
   }

   nir_push_else(b, nif);

   /* Otherwise, lower the texture instruction to read from 2D */
   assert(coord->num_components == 1 && "buffer textures are 1D");
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;

   nir_def *coord2d = coords_for_buffer_texture(b, coord);
   nir_instr_remove(&tex->instr);
   nir_builder_instr_insert(b, &tex->instr);
   nir_tex_instr_add_src(tex, nir_tex_src_backend1, coord2d);
   nir_steal_tex_src(tex, nir_tex_src_sampler_handle);
   nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   nir_block *else_block = nir_cursor_current_block(b->cursor);
   nir_pop_if(b, nif);

   /* Put it together with a phi */
   nir_def *phi = nir_if_phi(b, rgb32, &tex->def);
   nir_def_rewrite_uses(&tex->def, phi);
   nir_phi_instr *phi_instr = nir_def_as_phi(phi);
   nir_phi_src *else_src = nir_phi_get_src_from_block(phi_instr, else_block);
   nir_src_rewrite(&else_src->src, &tex->def);

   if (resident) {
      b->cursor = nir_after_phis(phi_instr->instr.block);
      nir_def *zero = nir_imm_zero(b, phi->num_components, phi->bit_size);
      nir_def *masked = nir_bcsel(b, resident, phi, zero);
      nir_def_rewrite_uses_after(phi, masked);
      phi = masked;
   }

   if (result)
      *result = phi;

   return true;
}

static bool
buffer_format_group_value_equal(nir_src a, nir_src b)
{
   if (nir_srcs_equal(a, b))
      return true;
   if (!nir_src_is_const(a) || !nir_src_is_const(b) ||
       a.ssa->bit_size != b.ssa->bit_size ||
       a.ssa->num_components != b.ssa->num_components)
      return false;

   const nir_const_value *av = nir_src_as_const_value(a);
   const nir_const_value *bv = nir_src_as_const_value(b);
   for (unsigned i = 0; i < a.ssa->num_components; ++i) {
      if (nir_const_value_as_uint(av[i], a.ssa->bit_size) !=
          nir_const_value_as_uint(bv[i], b.ssa->bit_size))
         return false;
   }
   return true;
}

static bool
buffer_format_group_source_equal(nir_src a, nir_src b)
{
   if (buffer_format_group_value_equal(a, b))
      return true;

   nir_intrinsic_instr *ai = nir_src_as_intrinsic(a);
   nir_intrinsic_instr *bi = nir_src_as_intrinsic(b);
   if (!ai || !bi || ai->intrinsic != nir_intrinsic_bindless_image_agx ||
       bi->intrinsic != nir_intrinsic_bindless_image_agx ||
       a.ssa->bit_size != b.ssa->bit_size ||
       a.ssa->num_components != b.ssa->num_components ||
       nir_intrinsic_desc_set(ai) != nir_intrinsic_desc_set(bi))
      return false;

   return buffer_format_group_value_equal(ai->src[0], bi->src[0]);
}

static bool
buffer_format_group_tex(nir_tex_instr *tex, nir_tex_instr *first)
{
   if (tex->op != nir_texop_txf || tex->sampler_dim != GLSL_SAMPLER_DIM_BUF ||
       tex->is_sparse || tex->num_srcs != first->num_srcs ||
       tex->dest_type != first->dest_type ||
       tex->def.num_components != first->def.num_components ||
       tex->def.bit_size != first->def.bit_size ||
       tex->texture_index != first->texture_index ||
       tex->sampler_index != first->sampler_index ||
       tex->backend_flags != first->backend_flags || tex->is_array ||
       tex->is_shadow || nir_tex_instr_src_index(tex, nir_tex_src_backend1) >= 0)
      return false;

   for (unsigned i = 0; i < tex->num_srcs; ++i) {
      if (tex->src[i].src_type != first->src[i].src_type)
         return false;
      if (tex->src[i].src_type != nir_tex_src_coord &&
          !buffer_format_group_source_equal(tex->src[i].src, first->src[i].src))
         return false;
   }

   return true;
}

struct buffer_format_group_sources {
   nir_tex_instr **tex;
   unsigned count;
};

static bool
buffer_format_group_src(nir_src *src, void *data)
{
   const struct buffer_format_group_sources *group = data;
   for (unsigned i = 0; i < group->count; ++i) {
      if (src->ssa == &group->tex[i]->def)
         return false;
   }
   return true;
}

static unsigned
lower_buffer_format_group(nir_shader *shader, nir_tex_instr *first,
                          const struct agx_texture_lower_options *options)
{
   nir_tex_instr *tex[32];
   unsigned count = 0;
   nir_instr *last = NULL;
   struct buffer_format_group_sources group = {tex, 0};

   for (nir_instr *instr = &first->instr; instr; instr = nir_instr_next(instr)) {
      if (!nir_foreach_src(instr, buffer_format_group_src, &group))
         break;

      if (instr->type == nir_instr_type_tex) {
         nir_tex_instr *current = nir_instr_as_tex(instr);
         if (!buffer_format_group_tex(current, first))
            break;
         tex[count++] = current;
         group.count = count;
         last = instr;
         if (count == ARRAY_SIZE(tex))
            break;
      } else if (instr->type != nir_instr_type_alu &&
                 instr->type != nir_instr_type_load_const &&
                 !(instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_bindless_image_agx)) {
         break;
      }
   }

   if (count < 4)
      return 0;

   for (nir_instr *instr = nir_instr_next(&first->instr); instr != last;) {
      nir_instr *next = nir_instr_next(instr);
      if (instr->type != nir_instr_type_tex)
         nir_instr_move(nir_before_instr(&first->instr), instr);
      instr = next;
   }

   nir_builder b = nir_builder_at(nir_before_instr(&first->instr));
   nir_def *desc = buffer_texture_descriptor_ptr(&b, first, options);
   nir_def *format = nir_ine_imm(&b, load_buffer_metadata(
      &b, desc, AGX_G17P_BUFFER_TEXTURE_RGB32_OFFSET, 32), 0);
   nir_def *values[2][32];
   nir_if *nif = nir_push_if(&b, format);
   nir_def *largest = first->src[nir_tex_instr_src_index(first, nir_tex_src_coord)].src.ssa;
   for (unsigned i = 1; i < count; ++i) {
      nir_def *coord = tex[i]->src[nir_tex_instr_src_index(tex[i], nir_tex_src_coord)].src.ssa;
      largest = nir_umax(&b, largest, coord);
   }
   nir_def *size = load_buffer_metadata(
      &b, desc, AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET, 32);
   nir_if *all = nir_push_if(&b, nir_ult(&b, largest, size));
   nir_def *rgb[2][32];
   for (unsigned path = 0; path < 2; ++path) {
      if (path)
         nir_push_else(&b, all);
      for (unsigned i = 0; i < count; ++i) {
         nir_tex_instr *copy = nir_instr_as_tex(nir_instr_clone(shader,
                                                               &tex[i]->instr));
         nir_builder_instr_insert(&b, &copy->instr);
         b.cursor = nir_before_instr(&copy->instr);
         lower_buffer_texture(&b, copy, true, options, 1, !path,
                              &rgb[path][i]);
      }
   }
   nir_pop_if(&b, all);
   for (unsigned i = 0; i < count; ++i)
      values[0][i] = nir_if_phi(&b, rgb[0][i], rgb[1][i]);
   nir_push_else(&b, nif);
   for (unsigned i = 0; i < count; ++i) {
      nir_tex_instr *copy = nir_instr_as_tex(nir_instr_clone(shader,
                                                            &tex[i]->instr));
      nir_builder_instr_insert(&b, &copy->instr);
      b.cursor = nir_before_instr(&copy->instr);
      lower_buffer_texture(&b, copy, true, options, 0, false, &values[1][i]);
   }
   nir_pop_if(&b, nif);

   for (unsigned i = 0; i < count; ++i) {
      nir_def *value = nir_if_phi(&b, values[0][i], values[1][i]);
      nir_def_rewrite_uses(&tex[i]->def, value);
      nir_instr_remove(&tex[i]->instr);
   }
   return count;
}

static bool
lower_buffer_format_groups(nir_shader *shader,
                           const struct agx_texture_lower_options *options)
{
   if (!options->gen9 || !options->buffer_metadata ||
       options->sparse_buffer_residency || shader->info.stage != MESA_SHADER_COMPUTE)
      return false;

   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      struct util_dynarray candidates;
      util_dynarray_init(&candidates, NULL);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (buffer_format_group_tex(tex, tex))
               util_dynarray_append(&candidates, tex);
         }
      }

      unsigned total = util_dynarray_num_elements(&candidates, nir_tex_instr *);
      for (unsigned i = 0; total >= 4 && i < total;) {
         nir_tex_instr *tex =
            *util_dynarray_element(&candidates, nir_tex_instr *, i);
         unsigned count = lower_buffer_format_group(shader, tex, options);
         if (count) {
            assert(count <= total - i);
            progress = true;
            nir_progress(true, impl, nir_metadata_none);
            i += count;
         } else {
            ++i;
         }
      }
      util_dynarray_fini(&candidates);
      nir_progress(progress, impl, nir_metadata_none);
   }

   return progress;
}

/* A texel offset counts as immediate when every component resolves to a
 * constant, including the per-component vector nir_lower_load_const_to_scalar
 * leaves behind. */
static bool
g17p_offset_is_constant(nir_def *offset)
{
   for (unsigned c = 0; c < offset->num_components; ++c) {
      if (!nir_scalar_is_const(nir_scalar_resolved(offset, c)))
         return false;
   }
   return true;
}

/* A signed x64 LOD in the 12-bit field of a G17P sampler bank-6 word, the
 * same bounds and truncation the immediate SMP LOD field uses. */
static nir_def *
pack_g17p_lod_field(nir_builder *b, nir_def *lod, float minimum, unsigned shift)
{
   nir_def *lod32 = nir_f2f32(b, lod);
   lod32 = nir_fmax(b, lod32, nir_imm_float(b, minimum));
   lod32 = nir_fmin(b, lod32, nir_imm_float(b, 2047.0f / 64.0f));
   nir_def *x64 = nir_f2i32(b, nir_fmul_imm(b, lod32, 64.0f));
   return nir_ishl_imm(b, nir_iand_imm(b, x64, 0xfff), shift);
}

/*
 * NIR indexes into array textures with unclamped floats (integer for txf). AGX
 * requires the index to be a clamped integer. Lower tex_src_coord into
 * tex_src_backend1 for array textures by type-converting and clamping.
 */
static bool
lower_regular_texture(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   const struct agx_texture_lower_options *options = data;
   const bool gen9 = options && options->gen9;
   const bool native_bounds = options && (options->gen9 || options->gen10);
   const bool g18g = options && options->g18g;
   const bool robust_texel_fetch = !options || options->robust_texel_fetch;
   const bool hw_txf_bounds = options && options->hw_txf_bounds;
   b->cursor = nir_before_instr(instr);

   if (nir_tex_instr_is_query(tex) && tex->op != nir_texop_lod)
      return false;

   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
      return lower_buffer_texture(b, tex, options && options->buffer_metadata,
                                  options, -1, false, NULL);
   }

   /* Don't lower twice */
   if (nir_tex_instr_src_index(tex, nir_tex_src_backend1) >= 0)
      return false;

   /* An implicit clamp uses the hardware bias + minimum form. Frontends
    * keeping sampler bias in the descriptor need a zero shader bias here.
    * Do this after constant folding, which turns txb(0) back into tex, and
    * immediately pack both sources below. */
   if (tex->op == nir_texop_tex &&
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0) {
      tex->op = nir_texop_txb;
      nir_tex_instr_add_src(tex, nir_tex_src_bias, nir_imm_float16(b, 0.0));
   }

   /* Get the coordinates */
   nir_def *coord = nir_steal_tex_src(tex, nir_tex_src_coord);
   nir_def *ms_idx = nir_steal_tex_src(tex, nir_tex_src_ms_index);

   /* Apply txf workaround, see libagx_lower_txf_robustness */
   bool is_txf = ((tex->op == nir_texop_txf) || (tex->op == nir_texop_txf_ms));
   nir_def *g17p_txf_oob = NULL;

   /* G17P's raw integer fetch is memory-safe but does not return the Vulkan
    * robustness value for every out-of-view coordinate. Without image
    * robustness an out-of-view fetch is undefined and a null descriptor
    * swizzles every channel to zero, so only robust images need this. The older workaround
    * below relies on that value: it moves an invalid layer or level to an
    * out-of-range coordinate and keeps the level itself, which the G17P
    * indexed fetch cannot even take as an immediate. Predicate the whole
    * access explicitly, fetch texel (0,0) of level 0 for an invalid one so the
    * hardware access itself stays inside the view, and select the
    * robustImageAccess2 value after the fetch. */
   if (native_bounds && robust_texel_fetch && tex->op == nir_texop_txf &&
       (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
        tex->sampler_dim == GLSL_SAMPLER_DIM_3D) &&
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0 &&
       !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)) {
      /* The size at the fetched level also counts array layers. A constant
       * zero component lies inside every valid level and layer, so only the
       * level check below can fail for it; a null descriptor already swizzles
       * every channel to zero. Skipping those components drops the size query,
       * which depends on a varying level, from a fetch of texel 0. */
      if (options->gen10) {
         nir_def *size = NULL;
         g17p_txf_oob = nir_imm_false(b);
         for (unsigned c = 0; c < tex->coord_components; ++c) {
            if (nir_scalar_is_zero(nir_get_scalar(coord, c)))
               continue;

            size = size ?: nir_get_texture_size(b, tex);
            g17p_txf_oob = nir_ior(
               b, g17p_txf_oob,
               nir_uge(b, nir_channel(b, coord, c), nir_channel(b, size, c)));
         }
      } else {
         /* Gen9 keeps the whole-vector check: the zero-component skip is
          * qualified on M5 only. */
         const unsigned n = tex->coord_components;
         nir_def *size = nir_trim_vector(b, nir_get_texture_size(b, tex), n);
         nir_def *checked = nir_trim_vector(b, coord, n);
         g17p_txf_oob = nir_bany(b, nir_uge(b, checked, size));
      }

      if (has_nonzero_lod(tex)) {
         /* The fetch consumes a 32-bit level, see the end of lower_texture().
          * Select it at that width. */
         nir_src *lod_src =
            &tex->src[nir_tex_instr_src_index(tex, nir_tex_src_lod)].src;
         nir_def *lod = nir_u2u32(b, lod_src->ssa);
         nir_def *levels = nir_build_texture_query(
            b, tex, nir_texop_query_levels, 1, nir_type_int32, false, false);
         g17p_txf_oob =
            nir_ior(b, g17p_txf_oob, nir_uge(b, lod, levels));
         nir_src_rewrite(lod_src,
                         nir_bcsel(b, g17p_txf_oob, nir_imm_int(b, 0), lod));
      }

      coord = nir_bcsel(b, g17p_txf_oob,
                        nir_imm_zero(b, coord->num_components,
                                     coord->bit_size),
                        coord);
   } else if (hw_txf_bounds && tex->op == nir_texop_txf &&
              (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
               tex->sampler_dim == GLSL_SAMPLER_DIM_3D) &&
              has_nonzero_lod(tex) &&
              nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0 &&
              !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)) {
      /* Spatial coordinates and saturated 16-bit array layers are bounded in
       * hardware, but an out-of-range LOD (including 65536/-65536 saturated
       * to 0xFFFF before 16-bit narrowing) must be checked against the view's
       * mip count and clamped to a safe hardware level. */
      nir_src *lod_src =
         &tex->src[nir_tex_instr_src_index(tex, nir_tex_src_lod)].src;
      nir_def *lod = nir_u2u32(b, lod_src->ssa);
      nir_def *levels = nir_build_texture_query(
         b, tex, nir_texop_query_levels, 1, nir_type_int32, false, false);
      g17p_txf_oob = nir_uge(b, lod, levels);
      nir_src_rewrite(lod_src,
                      nir_bcsel(b, g17p_txf_oob, nir_imm_int(b, 0), lod));
   } else if (hw_txf_bounds && tex->op == nir_texop_txf_ms && ms_idx &&
              !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)) {
      /* Check the saturated sample index against the view's sample count and
       * clamp invalid sample indices to 0 before packing so hardware never
       * fetches an invalid sample and robustImageAccess2 returns zero / alpha
       * one. */
      nir_def *sample = nir_u2u32(b, ms_idx);
      nir_def *samples = nir_build_texture_query(
         b, tex, nir_texop_texture_samples, 1, nir_type_int32, false, false);
      g17p_txf_oob = nir_uge(b, sample, samples);
      ms_idx = nir_bcsel(b, g17p_txf_oob, nir_imm_intN_t(b, 0, 16), ms_idx);
   }

   /* Without image robustness an out-of-range layer only has to stay memory
    * safe. The Gen9/10 texture unit bounds the 16-bit layer to the view
    * itself (G17G returns zero past the view's last layer, see the sample
    * layer clamp below), so the layer check is only needed for robust images
    * there. */
   const bool check_txf_layer =
      tex->is_array && (!native_bounds || robust_texel_fetch);

   if (is_txf && !g17p_txf_oob &&
       (has_nonzero_lod(tex) || check_txf_layer ||
        nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0) &&
       !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)) {

      int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_def *lod =
         lod_idx >= 0 ? tex->src[lod_idx].src.ssa : nir_undef(b, 1, 16);

      nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);

      /* libagx's integer-fetch robustness check consumes an integer mip
       * threshold, while the NIR source is floating point. */
      if (min_lod)
         min_lod = nir_f2u16(b, nir_fceil(b, min_lod));

      unsigned lidx = coord->num_components - 1;
      nir_def *layer = nir_channel(b, coord, lidx);

      /* Gen9/10 share the mip range, layout and layer-bound fields this
       * helper reads. Their width/height fields differ and are not read here.
       * Legacy mip fields overlap Gen10's sample/mipmap flags. */
      nir_def *(*lower)(nir_builder *, nir_def *, nir_def *, nir_def *,
                         nir_def *, nir_def *, nir_def *, nir_def *, nir_def *) =
         g18g            ? libagx_lower_txf_robustness_gen10
         : native_bounds ? libagx_lower_txf_robustness_gen9
                         : libagx_lower_txf_robustness;
      nir_def *replaced = lower(
         b, texture_descriptor_ptr(b, tex),
         nir_imm_bool(b, has_nonzero_lod(tex)), lod,
         nir_imm_bool(b, min_lod != NULL), min_lod ?: nir_undef(b, 1, 16),
         nir_imm_bool(b, check_txf_layer), layer, nir_channel(b, coord, 0));

      coord = nir_vector_insert_imm(b, coord, replaced, 0);
   }

   /* The layer is always the last component of the NIR coordinate, split it off
    * because we'll need to swizzle.
    */
   nir_def *layer = NULL;

   if (tex->is_array && tex->op != nir_texop_lod) {
      unsigned lidx = coord->num_components - 1;
      nir_def *unclamped_layer = nir_channel(b, coord, lidx);
      coord = nir_trim_vector(b, coord, lidx);

      /* G17P and G17G clamp the 16-bit hardware layer to the view in
       * hardware, so the software clamp to the layer count is dead weight
       * there. nir_get_texture_size() lowers to loading and DECODING the
       * packed descriptor -- once per indexed array sample -- purely to learn
       * the layer count, and on the CTS 4-lookup array-shadow shape that is
       * most of the shader: 1812 bytes with 17 device loads and ~79
       * iadd/isel, where Apple's compiler emits 316 bytes with ZERO device
       * loads from the same source. Dropping it takes that shader to 990.
       *
       * Verified on hardware with reproducers that read a distinct texel per
       * layer, so they detect which layer was sampled. G17P: 4- and 8-layer
       * images, requested layers 0/1/3/4/7/100/100000. G17G: a 12-layer 2D
       * array view, a 2-layer view at base layer 1, cube array views of 2
       * cubes and of 1 cube at base layer 6, layers 0..9/100/4095/65535/
       * 100000/1e9 and negative ones. All return the correctly clamped layer.
       * The clamp only sees the 16-bit layer, though: layer 65536 reads layer
       * 0, so saturate to 0xFFFF before the truncation. */
      const bool skip_layer_clamp = options && (options->gen9 || options->gen10);

      /* Round layer to nearest even */
      if (!is_txf) {
         unclamped_layer = nir_fround_even(b, unclamped_layer);

         /* Explicitly round negative to avoid undefined behaviour when constant
          * folding. This is load bearing on x86 builds.
          */
         unclamped_layer = nir_fmax(b, unclamped_layer, nir_imm_float(b, 0.0f));

         if (skip_layer_clamp) {
            unclamped_layer =
               nir_fmin(b, unclamped_layer, nir_imm_float(b, 65535.0f));
         }

         unclamped_layer = nir_f2u32(b, unclamped_layer);
      }

      /* For a cube array, the layer is zero-indexed component 3 of the
       * coordinate but the number of layers is component 2 of the txs result.
       */
      if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
         assert(lidx == 3 && "4 components");
         lidx = 2;
      }

      /* Clamp to max layer = (# of layers - 1) for out-of-bounds handling.
       * Layer must be 16-bits for the hardware, drop top bits after clamping.
       *
       * For txf, we drop out-of-bounds components rather than clamp, see the
       * above txf robustness workaround.
       */
      if (!(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP) && !is_txf &&
          !skip_layer_clamp) {
         nir_def *txs = nir_get_texture_size(b, tex);
         nir_def *nr_layers = nir_channel(b, txs, lidx);
         nir_def *max_layer = nir_iadd_imm(b, nr_layers, -1);
         layer = nir_umin(b, unclamped_layer, max_layer);
      } else if (is_txf && hw_txf_bounds &&
                 unclamped_layer->bit_size > 16 &&
                 !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)) {
         /* The hardware texture unit bounds 16-bit layers to the view, but
          * wider layers >= 65536 would truncate and wrap into range. Saturate
          * unsigned layers to 65535 before truncating to 16 bits so any
          * out-of-range layer stays out of range in hardware. */
         layer = nir_umin(
            b, unclamped_layer,
            nir_imm_intN_t(b, 65535, unclamped_layer->bit_size));
      } else {
         layer = unclamped_layer;
      }

      layer = nir_u2u16(b, layer);
   }

   /* Combine layer and multisample index into 32-bit so we don't need a vec5 or
    * vec6 16-bit coordinate tuple, which would be inconvenient in NIR for
    * little benefit (a minor optimization, I guess).
    */
   nir_def *sample_array = (ms_idx && layer)
                              ? nir_pack_32_2x16_split(b, ms_idx, layer)
                           : ms_idx ? nir_u2u32(b, ms_idx)
                           : layer  ? nir_u2u32(b, layer)
                                    : NULL;

   /* Combine into the final 32-bit tuple */
   if (sample_array != NULL) {
      unsigned end = coord->num_components;
      coord = nir_pad_vector(b, coord, end + 1);
      coord = nir_vector_insert_imm(b, coord, sample_array, end);
   }

   nir_tex_instr_add_src(tex, nir_tex_src_backend1, coord);

   /* G17P stages runtime offsets through sampler bank 6. Preserve a plain
    * component vector when the packer can publish X/Y with PACK_SMPIR. When
    * another operand shares that word, assemble its complete 32-bit value in
    * NIR and let the packer publish it once. Constants retain the compact
    * nibble-packed SMP immediate, and older generations retain their existing
    * packed register ABI. */
   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);

   if (offset != NULL) {
      /* G18G: native runtime gathers consume two signed halfword inputs.
       * Keep the pair together through register allocation; PACK_SMPIR
       * inserts its low nibbles and merges any array layer. Only constants
       * use the compact immediate-offset representation. */
      if (g18g && tex->op == nir_texop_tg4 &&
          !g17p_offset_is_constant(offset)) {
         assert(offset->num_components == 2);
         nir_def *packed = nir_pack_32_2x16_split(
            b, nir_i2i16(b, nir_channel(b, offset, 0)),
            nir_i2i16(b, nir_channel(b, offset, 1)));
         nir_tex_instr_add_src(tex, nir_tex_src_backend2, packed);
         goto offset_lowered;
      }

      /* G17G (generation 10) runtime offsets use the same PACK_SMPIR into
       * bank 6 as G17P, reading X and Y as separate 16-bit sources (native
       * texwit3/texwit4 d_auto, d_lvl2, d_bias1, d_grad, d_cmp*, d_gath*).
       * Keep the component vector, or the 2x16 pair for a gather, wherever
       * no array layer or register LOD/bias shares that bank. Those share
       * G17P's complete bank-6 word below: native G17G builds the same
       * layout (texwit3 d_lvlr: LOD f74=2 at bit 16, d_arr: layer f74=4 at
       * bit 0, X/Y f74=f76=5 at bits 12/28) with a register PACK_SMPIR
       * (f56=3) that the bank-6 PACK_SMPIR then merges (f51=2, f55=1). */
      if (options && options->gen10 && !g17p_offset_is_constant(offset) &&
          !tex->is_array && offset->num_components == 2) {
         const int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
         const int bias_idx = nir_tex_instr_src_index(tex, nir_tex_src_bias);
         const int min_idx = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
         const bool register_level =
            (lod_idx >= 0 && !nir_src_is_const(tex->src[lod_idx].src)) ||
            (bias_idx >= 0 && !nir_src_is_const(tex->src[bias_idx].src)) ||
            min_idx >= 0;
         if (tex->op == nir_texop_tg4) {
            nir_def *packed = nir_pack_32_2x16_split(
               b, nir_i2i16(b, nir_channel(b, offset, 0)),
               nir_i2i16(b, nir_channel(b, offset, 1)));
            nir_tex_instr_add_src(tex, nir_tex_src_backend2, packed);
            goto offset_lowered;
         }
         if (!register_level) {
            nir_tex_instr_add_src(tex, nir_tex_src_backend2, offset);
            goto offset_lowered;
         }
      }

      if ((gen9 || (options && options->gen10)) &&
          !g17p_offset_is_constant(offset)) {
         /* Gather's PACK_SMPIR form consumes X and Y as adjacent 16-bit
          * inputs.  Keep them in one 32-bit SSA value so register allocation
          * preserves that exact pair.  The generic runtime-offset path below
          * instead builds a complete bank-6 word for filtered samples; using
          * it for an array gather made the layer occupy X's low nibble and
          * produced wrong texels, while a plain gather arrived as two 32-bit
          * channels and was refused by the packer. */
         /* AGX3C (generation 10) stages an array gather's offset like an
          * array sample's: the layer word below (native texwit6 g_arr). */
         if (tex->op == nir_texop_tg4 &&
             !(options && options->gen10 && tex->is_array)) {
            assert(offset->num_components == 2);
            nir_def *packed = nir_pack_32_2x16_split(
               b, nir_i2i16(b, nir_channel(b, offset, 0)),
               nir_i2i16(b, nir_channel(b, offset, 1)));
            nir_tex_instr_add_src(tex, nir_tex_src_backend2, packed);
            goto offset_lowered;
         }

         /* Array layer, register LOD/bias, and a runtime 2D offset occupy
          * complementary fields of sampler bank 6. Multiple PACK_SMPIR
          * instructions replace that word instead of composing it, so publish
          * one already packed word: layer in bits 0..11, signed X/Y nibbles in
          * bits 12/28, and signed x64 LOD in bits 16..27. COPY can then move
          * the complete word to bank 6.
          *
          * Explicit levels are nonnegative; biases retain the full signed
          * 12-bit range.  These are the same bounds and truncation used by
          * the immediate SMP field.
          *
          * A sample with a minimum LOD keeps the same word, as Apple's
          * register packs do. A gradient's minimum takes bits 16..27. A bias
          * keeps them, and its minimum takes bits 0..11 unless an array
          * layer owns them; the 2D-array minimum is staged separately. */
         const int level_idx =
            nir_tex_instr_src_index(tex, nir_tex_src_lod);
         const int bias_idx =
            nir_tex_instr_src_index(tex, nir_tex_src_bias);
         const int min_lod_idx =
            nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
         const bool has_min_lod = min_lod_idx >= 0 &&
            nir_src_bit_size(tex->src[min_lod_idx].src) == 16;
         const int packed_level_idx =
            has_min_lod && tex->op == nir_texop_txd ? min_lod_idx
            : level_idx >= 0                        ? level_idx
                                                    : bias_idx;
         const bool pack_level = packed_level_idx >= 0 &&
            (has_min_lod ||
             !nir_src_is_const(tex->src[packed_level_idx].src)) &&
            nir_src_bit_size(tex->src[packed_level_idx].src) == 16;
         const bool pack_array_layer = tex->is_array && layer != NULL;
         const bool bias_minimum = has_min_lod && !tex->is_array &&
            tex->op == nir_texop_txb;

         /* A 3D bias with a minimum publishes the Z nibble in bits 0..3 of
          * the same word; its minimum goes to a packed register SMP names.
          * Apple's texture3d bias + min_lod_clamp + runtime offset packs the
          * X/Y nibbles and bias into one word (17055402030c1c10f08a0300),
          * then adds Z at bank-6 bit 0 (1705540600020010f0c20200). */
         if (offset->num_components == 3 && bias_minimum &&
             tex->sampler_dim == GLSL_SAMPLER_DIM_3D && pack_level) {
            nir_def *z = nir_iand_imm(b, nir_channel(b, offset, 2), 0xf);
            nir_def *x = nir_ishl_imm(
               b, nir_iand_imm(b, nir_channel(b, offset, 0), 0xf), 12);
            nir_def *y = nir_ishl_imm(
               b, nir_iand_imm(b, nir_channel(b, offset, 1), 0xf), 28);
            nir_def *packed = nir_ior(
               b, pack_g17p_lod_field(b, tex->src[bias_idx].src.ssa, -32.0f, 16),
               nir_ior(b, z, nir_ior(b, x, y)));
            nir_tex_instr_add_src(tex, nir_tex_src_backend2, packed);
            goto offset_lowered;
         }

         if (offset->num_components == 2 &&
             (pack_level || pack_array_layer)) {
            nir_def *packed =
               pack_array_layer
                  ? nir_iand_imm(b, nir_u2u32(b, layer), 0xfff)
               : bias_minimum
                  ? pack_g17p_lod_field(b, tex->src[min_lod_idx].src.ssa,
                                        0.0f, 0)
                  : nir_imm_int(b, 0);

            if (pack_level) {
               const bool signed_bias =
                  bias_idx >= 0 && packed_level_idx == bias_idx;
               packed = nir_ior(
                  b, packed,
                  pack_g17p_lod_field(b, tex->src[packed_level_idx].src.ssa,
                                      signed_bias ? -32.0f : 0.0f, 16));
            }

            nir_def *x =
               nir_ishl_imm(b,
                            nir_iand_imm(b, nir_channel(b, offset, 0), 0xf),
                            12);
            nir_def *y =
               nir_ishl_imm(b,
                            nir_iand_imm(b, nir_channel(b, offset, 1), 0xf),
                            28);
            packed = nir_ior(b, packed, nir_ior(b, x, y));
            nir_tex_instr_add_src(tex, nir_tex_src_backend2, packed);
            goto offset_lowered;
         }

         nir_tex_instr_add_src(tex, nir_tex_src_backend2, offset);
         goto offset_lowered;
      }

      nir_def *packed = NULL;

      for (unsigned c = 0; c < offset->num_components; ++c) {
         nir_def *nibble = nir_iand_imm(b, nir_channel(b, offset, c), 0xF);
         /* G17P runtime gathers feed two 16-bit inputs to PACK_SMPIR.
          * Keep each nibble in its own half, including constants which the
          * backend folds back to the compact SMP immediate. */
         const unsigned stride = gen9 && tex->op == nir_texop_tg4 ? 16 : 4;
         nir_def *shifted = nir_ishl_imm(b, nibble, stride * c);

         /* We pack with iadd instead of ior to let us fuse in the shift with an
          * iadd-lsl instruction.
          */
         if (packed != NULL)
            packed = nir_iadd(b, packed, shifted);
         else
            packed = shifted;
      }

      nir_tex_instr_add_src(tex, nir_tex_src_backend2, packed);
   }

offset_lowered:

   if (nir_tex_instr_src_index(tex, nir_tex_src_bias) >= 0 &&
       nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0) {

      nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
      nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
      nir_def *packed = nir_pack_32_2x16_split(b, bias, min_lod);
      nir_tex_instr_add_src(tex, nir_tex_src_lod_bias_min_agx, packed);
   }

   /* We reserve bound sampler #0 for texel fetch on every generation, so
    * offset API samplers by one. Bindless sampler handles already name their
    * heap entries directly and are left unchanged.
    */
   nir_def *sampler = nir_steal_tex_src(tex, nir_tex_src_sampler_offset);
   if (!sampler)
      sampler = nir_imm_intN_t(b, tex->sampler_index, 16);

   if (!is_txf &&
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) < 0) {
      nir_def *native_sampler =
         nir_iadd_imm(b, nir_u2u16(b, sampler), 1);

      /* HAL200's register/uniform sampler operand is the widened native-table
       * index. Immediate indices remain constants after folding, while a
       * dynamic API index must reach the backend as one 64-bit operand.
       * G18G's Gen10 register/uniform samplers use the same widened index. */
      if (gen9 || g18g)
         native_sampler = nir_u2u64(b, native_sampler);

      nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, native_sampler);
   }

   if (g17p_txf_oob) {
      b->cursor = nir_before_instr(instr);
      nir_def *oob_value = g17p_robust_oob_value(
         b, texture_descriptor_ptr(b, tex), tex->dest_type,
         tex->def.num_components, tex->def.bit_size);
      b->cursor = nir_after_instr(instr);
      nir_def *robust = nir_bcsel(b, g17p_txf_oob, oob_value, &tex->def);
      nir_def_rewrite_uses_after(&tex->def, robust);
   }

   return true;
}

static bool
legalize_image_lod(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   nir_src *src;

#define CASE(op, idx)                                                          \
   case nir_intrinsic_##op:                                                    \
   case nir_intrinsic_bindless_##op:                                           \
      src = &intr->src[idx];                                                   \
      break;

   switch (intr->intrinsic) {
      CASE(image_load, 3)
      CASE(image_sparse_load, 3)
      CASE(image_store, 4)
      CASE(image_size, 1)
   default:
      return false;
   }

#undef CASE

   if (src->ssa->bit_size == 16)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(src, nir_i2i16(b, src->ssa));
   return true;
}

static nir_def *
txs_for_image(nir_builder *b, nir_intrinsic_instr *intr,
              unsigned num_components, unsigned bit_size, bool query_samples)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   nir_def *lod = query_samples ? NULL : intr->src[1].ssa;
   nir_texop op = query_samples ? nir_texop_texture_samples : nir_texop_txs;

   nir_def *res =
      nir_build_tex(b, op, .texture_handle = intr->src[0].ssa, .lod = lod,
                    .dim = dim, .is_array = nir_intrinsic_image_array(intr),
                    .can_speculate = nir_instr_can_speculate(&intr->instr));

   /* Cube images are implemented as 2D arrays, so we need to divide here. */
   if (dim == GLSL_SAMPLER_DIM_CUBE && res->num_components > 2 &&
       !query_samples) {
      nir_def *divided = nir_udiv_imm(b, nir_channel(b, res, 2), 6);
      res = nir_vector_insert_imm(b, res, divided, 2);
   }

   return res;
}

static nir_def *
image_texel_address(nir_builder *b, nir_intrinsic_instr *intr,
                    bool return_index, bool gen9, bool gen10,
                    bool buffer_metadata, bool image_sideband)
{
   /* First, calculate the address of the PBE descriptor */
   nir_def *desc_address =
      nir_load_from_texture_handle_agx(b, intr->src[0].ssa);

   nir_def *coord = intr->src[1].ssa;

   /* For atomics, we always infer the format. Formatless multisampled image
    * stores use the return_index path, which only ignores the block size on
    * non-layered images (layered images divide layer_stride_sw by it).
    */
   enum pipe_format format = nir_intrinsic_format(intr);
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool layered = nir_intrinsic_image_array(intr) ||
                  (dim == GLSL_SAMPLER_DIM_CUBE) ||
                  (dim == GLSL_SAMPLER_DIM_3D);
   assert((return_index && !layered) || format != PIPE_FORMAT_NONE);

   nir_def *blocksize_B = nir_imm_int(b, util_format_get_blocksize(format));

   if (dim == GLSL_SAMPLER_DIM_BUF && return_index) {
      return nir_channel(b, coord, 0);
   } else if (dim == GLSL_SAMPLER_DIM_BUF && (buffer_metadata || gen9)) {
      /* Buffer atomics receive the PBE handle, but Gen9 has no AGX2
       * software address/offset fields. Use the same sideband as buffer
       * loads and stores, relative to the start of the descriptor. Gen10
       * Gallium has no such record: the PBE is a plain linear packet whose
       * buffer field names the texels, so it falls through to the generic
       * path below. */
      nir_def *desc = nir_iadd_imm(
         b, desc_address, -AGX_G17P_BUFFER_TEXTURE_PBE_OFFSET);
      nir_def *base = load_buffer_metadata(
         b, desc, AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET, 64);
      /* The base is the view's exact address: a texture tail offset counts
       * the view's own elements, which need not be the access's. */
      return nir_iadd(b, base,
                      nir_imul(b, nir_u2u64(b, nir_channel(b, coord, 0)),
                               nir_u2u64(b, blocksize_B)));
   } else if (dim == GLSL_SAMPLER_DIM_BUF && !(gen9 || gen10)) {
      return libagx_buffer_texel_address(b, desc_address, coord, blocksize_B);
   } else if (gen9 || gen10) {
      /* The software-metadata ABI keeps the addressing word in the storage
       * tail after the texture/PBE pair on both Gen9 and Gen10. Gallium's
       * plain Gen10 descriptors instead keep it in the PBE's last eight
       * bytes. Select the offset from the descriptor ABI, not just the GPU
       * generation. A caller without the metadata record whose storage
       * image descriptors still carry the tail
       * (AGX_NIR_TEXTURE_IMAGE_SIDEBAND) reads it there too; its texel
       * buffers keep the plain Gen10 form. */
      const bool sideband =
         buffer_metadata || (image_sideband && dim != GLSL_SAMPLER_DIM_BUF);
      nir_def *sw_address =
         nir_iadd_imm(b, desc_address,
                      gen10 && !sideband
                         ? AGX_GEN10_IMAGE_ATOMIC_SW_OFFSET
                         : AGX_G17P_IMAGE_ATOMIC_SW_FROM_PBE);
      if (dim == GLSL_SAMPLER_DIM_MS) {
         /* MS storage uses a linear PBE over the tiled allocation. Its
          * native width is the buffer window, not the source image width.
          * Reuse the eMRT store index and original aligned-width sideband. */
         nir_def *width_address = nir_iadd_imm(
            b, desc_address, AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET -
                                AGX_G17P_BUFFER_TEXTURE_PBE_OFFSET);
         nir_def *index = libagx_emrt_ms_index_gen9(
            b, sw_address, width_address, coord,
            nir_u2u32(b, intr->src[2].ssa), blocksize_B);
         if (return_index)
            return nir_u2u64(b, index);

         /* The linear Gen9 PBE stores address>>4 in the low 40 bits of
          * native words 2..3; the remaining bits are its row pitch. */
         nir_def *native23 = load_buffer_metadata(b, desc_address, 8, 64);
         nir_def *base = nir_ishl_imm(
            b, nir_iand_imm(b, native23, UINT64_C(0xffffffffff)), 4);
         return nir_iadd(
            b, base, nir_imul_imm(b, nir_u2u64(b, index),
                                  util_format_get_blocksize(format)));
      }
      return libagx_image_texel_address_gen9(
         b, desc_address, sw_address, coord, nir_u2u32(b, intr->src[2].ssa),
         blocksize_B, nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D),
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_MS), nir_imm_bool(b, layered),
         nir_imm_bool(b, return_index), nir_imm_bool(b, gen10));
   } else {
      return libagx_image_texel_address(
         b, desc_address, coord, nir_u2u32(b, intr->src[2].ssa), blocksize_B,
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_1D),
         nir_imm_bool(b, dim == GLSL_SAMPLER_DIM_MS), nir_imm_bool(b, layered),
         nir_imm_bool(b, return_index));
   }
}

static void
lower_buffer_image(nir_builder *b, nir_intrinsic_instr *intr,
                   bool buffer_metadata,
                   const struct agx_texture_lower_options *options)
{
   nir_def *coord_vector = intr->src[1].ssa;
   nir_def *coord = nir_channel(b, coord_vector, 0);
   /* The coordinate in the view, before the texture's tail offset */
   nir_def *view_coord = coord;
   nir_def *resident = NULL;

   /* If we're not bindless, assume we don't need an offset (GL driver) */
   if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {

      nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
      coord = buffer_metadata
         ? nir_iadd(b, coord, load_buffer_metadata(
              b, desc, AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET, 32))
         : libagx_buffer_texture_offset(b, desc, coord);

      /* imageLoad() reads the primary mapping, so an unbound page yields the
       * scratch page: mask it in software.  The hardware residency component
       * of a sparse load is no better here -- a texel buffer's descriptor is a
       * plain linear 2D texture, never AGX_IMAGE_MODE_SPARSE, so the unit has
       * nothing to report residency from. */
      if (buffer_metadata && options && options->sparse_buffer_residency)
         resident = buffer_texel_resident(b, desc, view_coord, options);
   } else if (intr->intrinsic == nir_intrinsic_bindless_image_store) {
      /* A store's handle addresses the PBE, which follows the native texture
       * at the start of the texel-buffer descriptor.
       */
      nir_def *desc = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
      coord = buffer_metadata
         ? nir_iadd(b, coord, load_buffer_metadata(
              b, desc, AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET -
                          AGX_G17P_BUFFER_TEXTURE_PBE_OFFSET, 32))
         : libagx_buffer_image_offset(b, desc, coord);
   }

   /* Lower the buffer load/store to a 2D image load/store, matching the 2D
    * texture/PBE descriptor the driver supplies for buffer images.
    */
   nir_def *coord2d = coords_for_buffer_texture(b, coord);
   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord2d, 4));
   if (buffer_metadata)
      nir_intrinsic_set_access(
         intr, nir_intrinsic_access(intr) | ACCESS_BUFFER_AGX);
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);

   if (resident) {
      const bool sparse =
         intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;
      /* A sparse load's last component is the residency code, which the
       * hardware fills in from a descriptor that knows nothing about this
       * buffer's residency.  Replace it; zero the data either way. */
      const unsigned data = intr->def.num_components - (sparse ? 1 : 0);

      b->cursor = nir_after_instr(&intr->instr);

      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      nir_def *zero = nir_imm_zero(b, 1, intr->def.bit_size);
      for (unsigned i = 0; i < data; ++i) {
         comps[i] =
            nir_bcsel(b, resident, nir_channel(b, &intr->def, i), zero);
      }
      if (sparse) {
         comps[data] = nir_u2uN(
            b,
            nir_bcsel(b, resident, nir_imm_int(b, AGX_RESIDENT),
                      nir_imm_int(b, AGX_RESIDENT ^ 1)),
            intr->def.bit_size);
      }

      nir_def_rewrite_uses_after(&intr->def,
                                 nir_vec(b, comps, intr->def.num_components));
   }
}

static void
lower_1d_image(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *coord = intr->src[1].ssa;
   bool is_array = nir_intrinsic_image_array(intr);
   nir_def *zero = nir_imm_intN_t(b, 0, coord->bit_size);

   if (is_array) {
      assert(coord->num_components >= 2);
      coord =
         nir_vec3(b, nir_channel(b, coord, 0), zero, nir_channel(b, coord, 1));
   } else {
      assert(coord->num_components >= 1);
      coord = nir_vec2(b, coord, zero);
   }

   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord, 4));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
}

/*
 * Just like for txf, we need special handling around layers (and LODs, but we
 * don't support mipmapped images yet) for robust image_loads. See
 * libagx_lower_txf_robustness for more info.
 */
static bool
lower_image_load_robustness(nir_builder *b, nir_intrinsic_instr *intr,
                            bool gen9, bool g18g)
{
   if (nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS)
      return false;

   /* We only need to worry about array-like loads */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   if (!nir_intrinsic_image_array(intr) && dim != GLSL_SAMPLER_DIM_CUBE)
      return false;

   /* Determine the coordinate component of the layer. Cubes and cube arrays
    * keep their array in their last non-array coordinate component, other
    * arrays are immediately after.
    */
   unsigned lidx = glsl_get_sampler_dim_coordinate_components(dim);
   if (dim == GLSL_SAMPLER_DIM_CUBE)
      lidx--;

   nir_def *coord = intr->src[1].ssa;
   nir_def *lod = nir_undef(b, 1, 16);
   nir_def *layer = nir_channel(b, coord, lidx);

   /* image_load is effectively the same as txf, reuse the txf lower */
   nir_def *(*lower)(nir_builder *, nir_def *, nir_def *, nir_def *,
                      nir_def *, nir_def *, nir_def *, nir_def *, nir_def *) =
      g18g    ? libagx_lower_txf_robustness_gen10
      : gen9  ? libagx_lower_txf_robustness_gen9
              : libagx_lower_txf_robustness;
   nir_def *replaced = lower(
      b, nir_load_from_texture_handle_agx(b, intr->src[0].ssa),
      nir_imm_bool(b, false /* lower LOD */), lod,
      nir_imm_bool(b, false /* lower min LOD */), lod,
      nir_imm_bool(b, true /* lower layer */), layer, nir_channel(b, coord, 0));

   nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, replaced, 0));
   return true;
}

/*
 * A formatless load's robustImageAccess2 value depends on the bound view: zero,
 * with a one alpha when its format has none. The common robust-access lowering
 * can only fold that from the intrinsic's format, so the driver leaves G17P
 * formatless loads to this pass. Predicate the load against the image size,
 * read texel 0 for an invalid coordinate so the native fetch stays inside the
 * image, and select the view's out-of-bounds value after the fetch.
 */
static void
lower_g17p_formatless_image_load(nir_builder *b, nir_intrinsic_instr *intr)
{
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool is_array = nir_intrinsic_image_array(intr);
   const unsigned num_coords = nir_image_intrinsic_coord_components(intr);
   const bool cube = dim == GLSL_SAMPLER_DIM_CUBE;

   /* The size of a cube is that of one face, and a cube array counts cubes. */
   nir_def *size =
      nir_build_tex(b, nir_texop_txs, .texture_handle = intr->src[0].ssa,
                    .lod = nir_imm_int(b, 0), .dim = dim, .is_array = is_array,
                    .can_speculate = nir_instr_can_speculate(&intr->instr));
   if (cube) {
      nir_def *layers = is_array ? nir_channel(b, size, 2) : nir_imm_int(b, 6);
      size = nir_vec3(b, nir_channel(b, size, 0), nir_channel(b, size, 1),
                      layers);
   }

   nir_def *coord = intr->src[1].ssa;
   nir_def *oob = nir_bany(b, nir_uge(b, nir_trim_vector(b, coord, num_coords),
                                      nir_trim_vector(b, size, num_coords)));
   nir_src_rewrite(&intr->src[1],
                   nir_bcsel(b, oob,
                             nir_imm_zero(b, coord->num_components,
                                          coord->bit_size),
                             coord));

   nir_def *oob_value = g17p_robust_oob_value(
      b, nir_load_from_texture_handle_agx(b, intr->src[0].ssa),
      nir_intrinsic_dest_type(intr), intr->def.num_components,
      intr->def.bit_size);

   b->cursor = nir_after_instr(&intr->instr);
   nir_def_rewrite_uses_after(&intr->def,
                              nir_bcsel(b, oob, oob_value, &intr->def));
}

nir_alu_type
agx_image_store_base_type(const nir_intrinsic_instr *intr)
{
   nir_alu_type base =
      nir_alu_type_get_base_type(nir_intrinsic_src_type(intr));
   if (base != nir_type_invalid)
      return base;

   enum pipe_format fmt = nir_intrinsic_format(intr);
   if (fmt == PIPE_FORMAT_NONE)
      return nir_type_invalid;

   if (util_format_is_pure_sint(fmt))
      return nir_type_int;
   if (util_format_is_pure_uint(fmt))
      return nir_type_uint;
   return nir_type_float;
}

static bool
lower_images(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   const struct agx_texture_lower_options *options = data;
   const bool gen9 = options && options->gen9;
   const bool g18g = options && options->g18g;

   switch (intr->intrinsic) {
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_bindless_image_store: {
      /* Legalize MSAA index */
      nir_src_rewrite(&intr->src[2], nir_u2u16(b, intr->src[2].ssa));

      /* AGX3's imgwci storage image store encoding requires 32-bit data
       * channels. Widen sub-32-bit values (such as lowp/mediump spilled MRT
       * epilog outputs on M5 or mediump imageStore() calls) here so every
       * AGX3 image store path gets 32-bit data uniformly. */
      if ((gen9 || (options && options->gen10)) &&
          (intr->intrinsic == nir_intrinsic_image_store ||
           intr->intrinsic == nir_intrinsic_bindless_image_store) &&
          intr->src[3].ssa->bit_size < 32) {
         nir_alu_type base = agx_image_store_base_type(intr);
         /* Every valid NIR image_store producer either sets src_type (SPIR-V,
          * GLSL, tilebuffer lowering) or supplies a non-NONE pipe_format. */
         assert(base == nir_type_float || base == nir_type_int ||
                base == nir_type_uint);
         if (base == nir_type_float || base == nir_type_int ||
             base == nir_type_uint) {
            nir_def *value =
               nir_convert_to_bit_size(b, intr->src[3].ssa, base, 32);
            nir_src_rewrite(&intr->src[3], value);
            nir_intrinsic_set_src_type(intr, base | 32);
         }
      }

      if (gen9 && options->robust_formatless_images &&
          intr->intrinsic == nir_intrinsic_bindless_image_load &&
          nir_intrinsic_format(intr) == PIPE_FORMAT_NONE &&
          !(nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS) &&
          nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_MS &&
          nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF &&
          intr->def.num_components == 4) {
         lower_g17p_formatless_image_load(b, intr);
         b->cursor = nir_before_instr(&intr->instr);
      } else if (intr->intrinsic == nir_intrinsic_image_load ||
                 intr->intrinsic == nir_intrinsic_bindless_image_load ||
                 intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {
         lower_image_load_robustness(b, intr,
                                      gen9 || (options && options->gen10),
                                      g18g);
      }

      switch (nir_intrinsic_image_dim(intr)) {
      case GLSL_SAMPLER_DIM_1D:
         lower_1d_image(b, intr);
         return true;

      case GLSL_SAMPLER_DIM_BUF:
         lower_buffer_image(b, intr,
                            gen9 || (options && options->buffer_metadata),
                            options);
         return true;

      default:
         return true;
      }
   }

   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_bindless_image_samples:
      nir_def_rewrite_uses(
         &intr->def,
         txs_for_image(
            b, intr, intr->def.num_components, intr->def.bit_size,
            intr->intrinsic == nir_intrinsic_bindless_image_samples));
      return true;

   case nir_intrinsic_bindless_image_texel_address:
      nir_def_rewrite_uses(&intr->def,
                           image_texel_address(b, intr, false, gen9,
                                               options && options->gen10,
                                               options && options->buffer_metadata,
                                               options && options->image_sideband));
      return true;

   case nir_intrinsic_is_sparse_texels_resident:
      /* Residency information is in bit 0, so we need to mask. Unclear what's
       * in the upper bits. For now, let's match the blob.
       */
      nir_def_replace(
         &intr->def,
         nir_ieq_imm(b, nir_iand_imm(b, intr->src[0].ssa, 1), AGX_RESIDENT));
      return true;

   case nir_intrinsic_sparse_residency_code_and:
      /* ior because residency codes are inverted from NIR */
      nir_def_replace(&intr->def,
                      nir_ior(b, intr->src[0].ssa, intr->src[1].ssa));
      return true;

   case nir_intrinsic_image_size:
   case nir_intrinsic_image_texel_address:
      UNREACHABLE("should've been lowered");

   default:
      return false;
   }
}

/*
 * Map out-of-bounds storage texel buffer accesses and multisampled image stores
 * to -1 indices, which will become an out-of-bounds hardware access. This gives
 * cheap robustness2.
 */
static bool
lower_robustness(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
      break;
   default:
      return false;
   }

   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool array = nir_intrinsic_image_array(intr);
   unsigned size_components = nir_image_intrinsic_coord_components(intr);
   /* G17P needs an explicit bounds predicate to make ordinary 2D image loads
    * return zero.  Its ordinary 2D stores are already bounds-safe, and adding
    * imageSize() to them leaves a redundant destination-descriptor crawl that
    * prevents the native sampled-image-to-storage-image path from matching.
    * Sparse-load residency semantics are not covered by this lowering. */
   const bool lower_2d = data && *(const bool *)data &&
                         intr->intrinsic == nir_intrinsic_image_deref_load &&
                         dim == GLSL_SAMPLER_DIM_2D && !array;

   if (lower_2d)
      size_components = 2;

   nir_def *deref = intr->src[0].ssa;
   nir_def *coord = intr->src[1].ssa;

   if (!lower_2d && dim != GLSL_SAMPLER_DIM_BUF &&
       !(dim == GLSL_SAMPLER_DIM_MS &&
         intr->intrinsic == nir_intrinsic_image_deref_store))
      return false;

   /* Bounds check the coordinate */
   nir_def *size =
      nir_image_deref_size(b, size_components, 32, deref, nir_imm_int(b, 0),
                           .image_dim = dim, .image_array = array);
   nir_def *checked_coord =
      lower_2d ? nir_trim_vector(b, coord, size_components) : coord;
   nir_def *oob = nir_bany(b, nir_uge(b, checked_coord, size));

   /* Bounds check the sample */
   if (dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_image_deref_samples(b, 32, deref, .image_dim = dim,
                                                 .image_array = array);

      oob = nir_ior(b, oob, nir_uge(b, intr->src[2].ssa, samples));
   }

   /* Replace the last coordinate component with a large coordinate for
    * out-of-bounds. We pick 0xFFF0 as it fits in 16-bit, and it is not signed
    * as 32-bit so we won't get in-bounds coordinates for arrays due to two's
    * complement wraparound. Additionally it still meets this requirement after
    * adding 0xF, the maximum tail offset.
    *
    * This ensures the resulting hardware coordinate is definitely
    * out-of-bounds, giving hardware-level robustness2 behaviour.
    */
   unsigned c = size_components - 1;
   nir_def *r =
      nir_bcsel(b, oob, nir_imm_int(b, 0xFFF0), nir_channel(b, coord, c));

   nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, r, c));

   /* G17P does not return zero for every far-out coordinate encoding.  Keep
    * the hardware access itself in-bounds-safe, then make robustness2's load
    * value explicit from the same bounds predicate. */
   if (lower_2d && intr->intrinsic == nir_intrinsic_image_deref_load) {
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *robust = nir_bcsel(
         b, oob, nir_imm_zero(b, intr->def.num_components,
                              intr->def.bit_size),
         &intr->def);
      nir_def_rewrite_uses_after(&intr->def, robust);
   }

   return true;
}

/*
 * Early texture lowering passes, called by the driver before lowering
 * descriptor bindings. That means these passes operate on texture derefs. The
 * purpose is to make descriptor crawls explicit in the NIR, so that the driver
 * can accurately lower descriptors after this pass but before calling
 * the full agx_nir_lower_texture.
 */
static bool lower_texture_early(nir_shader *s, bool support_lod_bias,
                                bool lower_2d_robustness,
                                bool shadow_keeps_sampler_bias);

bool
agx_nir_lower_texture_early(nir_shader *s, bool support_lod_bias)
{
   return lower_texture_early(s, support_lod_bias, false, false);
}

static bool
lower_g17p_3d_runtime_offset(const nir_instr *instr, UNUSED const void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   const nir_tex_instr *tex = nir_instr_as_tex(instr);
   const int offset =
      nir_tex_instr_src_index(tex, nir_tex_src_offset);
   /* Coordinate addition scales the offset by the base level, which is only
    * right when the base level is sampled. A bias with a minimum LOD, which
    * selects other levels, keeps its native staged offset; an immediate
    * offset keeps SMP's. */
   return tex->sampler_dim == GLSL_SAMPLER_DIM_3D && offset >= 0 &&
          !g17p_offset_is_constant(tex->src[offset].src.ssa) &&
          !(tex->op == nir_texop_txb &&
            nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0 &&
            nir_tex_instr_src_index(tex, nir_tex_src_projector) < 0);
}

enum g17p_3d_runtime_offset_mode {
   G17P_3D_RUNTIME_OFFSET = BITFIELD_BIT(0),
   G17P_PROJECTED_3D_RUNTIME_OFFSET = BITFIELD_BIT(1),
};

static unsigned
g17p_3d_runtime_offset_modes(const nir_shader *s)
{
   unsigned modes = 0;

   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (!lower_g17p_3d_runtime_offset(instr, NULL))
               continue;

            const nir_tex_instr *tex = nir_instr_as_tex(instr);
            modes |= G17P_3D_RUNTIME_OFFSET;
            if (nir_tex_instr_src_index(tex, nir_tex_src_coord) >= 0 &&
                nir_tex_instr_src_index(tex, nir_tex_src_projector) >= 0)
               modes |= G17P_PROJECTED_3D_RUNTIME_OFFSET;
         }
      }
   }

   return modes;
}

/* Compose a projected 3D runtime offset in homogeneous coordinates before
 * the projective divide:
 *
 *   (coord + offset * (q / size)) / q == coord / q + offset / size
 *
 * This preserves the API's project-then-offset value while avoiding a second
 * add after the projected coordinate has already been quantized. The FFMA
 * form has the same arithmetic instruction count as lowering projection and
 * offset independently. */
static bool
lower_g17p_projected_3d_runtime_offset(nir_builder *b, nir_instr *instr,
                                       UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int projector_idx =
      nir_tex_instr_src_index(tex, nir_tex_src_projector);
   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_3D || coord_idx < 0 ||
       projector_idx < 0 || offset_idx < 0 ||
       g17p_offset_is_constant(tex->src[offset_idx].src.ssa))
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   nir_def *projector =
      nir_steal_tex_src(tex, nir_tex_src_projector);
   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);
   nir_def *size_rcp = nir_frcp(b, nir_i2f32(b,
      nir_get_texture_size(b, tex)));
   nir_def *homogeneous_scale = nir_fmul(b, size_rcp, projector);
   nir_def *homogeneous =
      nir_ffma(b, nir_i2f32(b, offset), homogeneous_scale, coord);
   nir_src_rewrite(&tex->src[coord_idx].src,
                   nir_fmul(b, homogeneous, nir_frcp(b, projector)));
   return true;
}

/* nir_lower_tex intentionally leaves rectangle samples implicit because
 * they have no mip chain. G17P still requires an explicit level outside a
 * derivative-capable stage. The sampler retains unnormalized coordinates;
 * this changes only the LOD operand, just like ordinary non-fragment tex.
 * Run after sampler-bias lowering, whose bias is ignored in these stages.
 */
static bool
lower_g17p_rectangle_lod(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (nir_shader_supports_implicit_lod(b->shader) ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_RECT ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb))
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_steal_tex_src(tex, nir_tex_src_bias);
   tex->op = nir_texop_txl;
   nir_tex_instr_add_src(tex, nir_tex_src_lod, nir_imm_float(b, 0.0));
   return true;
}

bool
agx_nir_lower_g17p_texture_early(nir_shader *s, bool support_lod_bias,
                                 bool sampler_has_lod_bias)
{
   bool progress = false;
   const unsigned runtime_offset_modes =
      g17p_3d_runtime_offset_modes(s);

   /* HAL200's texture_load has no witnessed integer-offset form.  A txf
    * offset is defined as integer coordinate addition, so lower it before
    * descriptor binding instead of admitting an unencodable backend source. */
   NIR_PASS(progress, s, nir_lower_tex,
            &(nir_lower_tex_options){.lower_txf_offset = true});
   if (runtime_offset_modes & G17P_PROJECTED_3D_RUNTIME_OFFSET) {
      NIR_PASS(progress, s, nir_shader_instructions_pass,
               lower_g17p_projected_3d_runtime_offset,
               nir_metadata_control_flow, NULL);
   }
   /* A depth comparison keeps its sampler's bias in the descriptor: the
    * AGX3B comparison sample has no witnessed per-lane level form in the
    * vertex stage, where a GLSL `texture()` arrives as an explicit level
    * zero, and folding a bias into that level would leave the shape
    * unencodable. agx_upload_samplers() packs the field to match. A Gen4
    * sampler has no bias field to keep it in, so there the shader applies
    * it like any other bias. */
   progress |= lower_texture_early(s, support_lod_bias, false,
                                   sampler_has_lod_bias);
   NIR_PASS(progress, s, nir_shader_tex_pass, lower_g17p_rectangle_lod,
            nir_metadata_control_flow, NULL);

   /* No qualified runtime-offset ABI can place 3D Z beside the gradient
    * banks, and the inferred bank-8 continuation fails filtered samples on
    * hardware. Lower a dynamic 3D offset to normalized coordinate addition
    * after the common early pass has projected coordinates. The witnessed
    * immediate-offset form remains unchanged. */
   if (runtime_offset_modes & G17P_3D_RUNTIME_OFFSET) {
      NIR_PASS(progress, s, nir_lower_tex,
               &(nir_lower_tex_options){
                  .lower_offset_filter = lower_g17p_3d_runtime_offset,
               });
   }
   return progress;
}

static bool
lower_agx3d_implicit_min_lod(nir_builder *b, nir_tex_instr *tex,
                             UNUSED void *data)
{
   if ((tex->op != nir_texop_tex && tex->op != nir_texop_txb) ||
       !nir_shader_supports_implicit_lod(b->shader))
      return false;

   if (nir_tex_instr_src_index(tex, nir_tex_src_min_lod) < 0)
      return false;

   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   assert(coord_idx >= 0);

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *lod;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_RECT) {
      lod = nir_imm_float(b, 0.0f);
   } else {
      nir_def *coord = tex->src[coord_idx].src.ssa;
      unsigned spatial_components =
         tex->coord_components - (tex->is_array ? 1 : 0);
      nir_def *spatial = nir_trim_vector(b, coord, spatial_components);
      nir_def *dPdx = nir_ddx(b, spatial);
      nir_def *dPdy = nir_ddy(b, spatial);

      if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
         nir_def *size = nir_i2f32(b, nir_get_texture_size(b, tex));
         nir_def *abs_p = nir_fabs(b, spatial);
         nir_def *abs_x = nir_channel(b, abs_p, 0);
         nir_def *abs_y = nir_channel(b, abs_p, 1);
         nir_def *abs_z = nir_channel(b, abs_p, 2);

         nir_def *cond_z = nir_fge(b, abs_z, nir_fmax(b, abs_x, abs_y));
         nir_def *cond_y = nir_fge(b, abs_y, nir_fmax(b, abs_x, abs_z));

         unsigned yzx[3] = {1, 2, 0};
         unsigned xzy[3] = {0, 2, 1};

         nir_def *Q = nir_bcsel(
            b, cond_z, spatial,
            nir_bcsel(b, cond_y, nir_swizzle(b, spatial, xzy, 3),
                      nir_swizzle(b, spatial, yzx, 3)));
         nir_def *dQdx = nir_bcsel(
            b, cond_z, dPdx,
            nir_bcsel(b, cond_y, nir_swizzle(b, dPdx, xzy, 3),
                      nir_swizzle(b, dPdx, yzx, 3)));
         nir_def *dQdy = nir_bcsel(
            b, cond_z, dPdy,
            nir_bcsel(b, cond_y, nir_swizzle(b, dPdy, xzy, 3),
                      nir_swizzle(b, dPdy, yzx, 3)));

         nir_def *rcp_Q_z = nir_frcp(b, nir_channel(b, Q, 2));
         nir_def *tmp = nir_fmul(b, nir_trim_vector(b, Q, 2), rcp_Q_z);
         nir_def *dx = nir_fmul(
            b, rcp_Q_z,
            nir_fsub(b, nir_trim_vector(b, dQdx, 2),
                     nir_fmul(b, tmp, nir_channel(b, dQdx, 2))));
         nir_def *dy = nir_fmul(
            b, rcp_Q_z,
            nir_fsub(b, nir_trim_vector(b, dQdy, 2),
                     nir_fmul(b, tmp, nir_channel(b, dQdy, 2))));

         nir_def *M = nir_fmax(b, nir_fdot(b, dx, dx), nir_fdot(b, dy, dy));
         nir_def *L = nir_channel(b, size, 0);
         lod = nir_fadd_imm(
            b,
            nir_fmul_imm(b, nir_flog2(b, nir_fmul(b, L, nir_fmul(b, L, M))),
                         0.5f),
            -1.0f);
      } else {
         unsigned component_mask =
            tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? 7 : 3;
         nir_def *size = nir_channels(
            b, nir_i2f32(b, nir_get_texture_size(b, tex)), component_mask);
         nir_def *scaled_dx = nir_fmul(b, dPdx, size);
         nir_def *scaled_dy = nir_fmul(b, dPdy, size);
         nir_def *rho = nir_fmax(
            b, nir_fsqrt(b, nir_fdot(b, scaled_dx, scaled_dx)),
            nir_fsqrt(b, nir_fdot(b, scaled_dy, scaled_dy)));
         lod = nir_flog2(b, rho);
      }
   }

   if (tex->op == nir_texop_txb) {
      nir_def *bias = nir_steal_tex_src(tex, nir_tex_src_bias);
      assert(bias != NULL);
      lod = nir_fadd(b, lod, nir_f2f32(b, bias));
   }

   nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
   assert(min_lod != NULL);
   lod = nir_fmax(b, lod, nir_f2f32(b, min_lod));

   tex->op = nir_texop_txl;
   nir_tex_instr_add_src(tex, nir_tex_src_lod, lod);
   return true;
}

static nir_def *
agx3d_get_texture_levels(nir_builder *b, nir_tex_instr *tex)
{
   unsigned num_srcs = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle)
         num_srcs++;
   }

   nir_tex_instr *query = nir_tex_instr_create(b->shader, num_srcs);
   query->op = nir_texop_query_levels;
   query->sampler_dim = tex->sampler_dim;
   query->is_array = tex->is_array;
   query->texture_index = tex->texture_index;
   query->texture_non_uniform = tex->texture_non_uniform;
   query->dest_type = nir_type_int32;

   unsigned dst = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_texture_deref ||
          tex->src[i].src_type == nir_tex_src_texture_offset ||
          tex->src[i].src_type == nir_tex_src_texture_handle) {
         query->src[dst].src_type = tex->src[i].src_type;
         query->src[dst].src = nir_src_for_ssa(tex->src[i].src.ssa);
         dst++;
      }
   }

   nir_def_init(&query->instr, &query->def, 1, 32);
   nir_builder_instr_insert(b, &query->instr);
   return &query->def;
}

static bool
lower_agx3d_txl_runtime_offset(nir_builder *b, nir_tex_instr *tex,
                               UNUSED void *data)
{
   if (tex->op != nir_texop_txl || tex->sampler_dim == GLSL_SAMPLER_DIM_RECT)
      return false;

   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (offset_idx < 0 || g17p_offset_is_constant(tex->src[offset_idx].src.ssa))
      return false;

   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (coord_idx < 0 || lod_idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *coord = tex->src[coord_idx].src.ssa;
   nir_def *lod = nir_f2f32(b, tex->src[lod_idx].src.ssa);
   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);

   unsigned spatial_components =
      tex->coord_components - (tex->is_array ? 1 : 0);
   nir_def *txs0 =
      nir_trim_vector(b, nir_get_texture_size(b, tex), spatial_components);
   nir_def *levels = agx3d_get_texture_levels(b, tex);
   nir_def *max_level = nir_imax(b, nir_iadd_imm(b, levels, -1), nir_imm_int(b, 0));

   /* Vulkan mip level selection for nearest-mipmap: clamp(ceil(lod + 0.5) - 1, 0, max_level). */
   nir_def *level = nir_f2i32(
      b, nir_fmax(b, nir_fadd_imm(b, nir_fceil(b, nir_fadd_imm(b, lod, 0.5f)), -1.0f),
                  nir_imm_float(b, 0.0f)));
   level = nir_imin(b, level, max_level);

   nir_def *size = nir_imax(b, nir_ushr(b, txs0, level), nir_imm_int(b, 1));
   nir_def *scale = nir_frcp(b, nir_i2f32(b, size));
   nir_def *spatial = nir_trim_vector(b, coord, spatial_components);
   nir_def *offset_spatial =
      nir_fadd(b, spatial, nir_fmul(b, nir_i2f32(b, offset), scale));

   nir_def *new_coord = offset_spatial;
   if (tex->is_array) {
      unsigned a = tex->coord_components - 1;
      new_coord = nir_pad_vector(b, offset_spatial, tex->coord_components);
      new_coord =
         nir_vector_insert_imm(b, new_coord, nir_channel(b, coord, a), a);
   }

   nir_src_rewrite(&tex->src[coord_idx].src, new_coord);
   return true;
}

static bool
lower_agx3d_runtime_offset(const nir_instr *instr, UNUSED const void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   const nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op == nir_texop_tg4 || tex->op == nir_texop_txf)
      return false;

   const int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   return offset >= 0 && !g17p_offset_is_constant(tex->src[offset].src.ssa);
}

/*
 * AGX3D early texture lowering: runs the common AGX3 early pass, then lowers
 * texture forms that AGX3D does not encode directly in hardware SMP:
 *  - min_lod on implicit/biased/gradient samples (lowered to explicit LOD)
 *  - 3D and shadow gradient samples (lowered to explicit LOD)
 *  - non-gather runtime texel offsets (folded into normalized coordinates,
 *    scaled by the sampled mip level for explicit LOD)
 */
bool
agx_nir_lower_agx3d_texture_early(nir_shader *s, bool support_lod_bias,
                                  bool sampler_has_lod_bias)
{
   bool progress =
      agx_nir_lower_g17p_texture_early(s, support_lod_bias,
                                       sampler_has_lod_bias);

   NIR_PASS(progress, s, nir_shader_tex_pass, lower_agx3d_implicit_min_lod,
            nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_lower_tex,
            &(nir_lower_tex_options){
               .lower_txd_clamp = true,
               .lower_txd_3d = true,
               .lower_txd_shadow = true,
               .lower_txd_cube_map = true,
            });

   NIR_PASS(progress, s, nir_shader_tex_pass, lower_agx3d_txl_runtime_offset,
            nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_lower_tex,
            &(nir_lower_tex_options){
               .lower_offset_filter = lower_agx3d_runtime_offset,
            });

   return progress;
}

static bool
lower_texture_early(nir_shader *s, bool support_lod_bias,
                    bool lower_2d_robustness, bool shadow_keeps_sampler_bias)
{
   bool progress = false;

   NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_robustness,
            nir_metadata_control_flow, &lower_2d_robustness);

   nir_lower_tex_options lower_tex_options = {
      .lower_txp = ~0,
      .lower_invalid_implicit_lod = true,
      .lower_tg4_offsets = true,
      .lower_index_to_offset = true,
      .lower_sampler_lod_bias = support_lod_bias,
      .keep_sampler_lod_bias_for_shadow = shadow_keeps_sampler_bias,

      /* Unclear if/how mipmapped 1D textures work in the hardware. */
      .lower_1d = true,

      /* XXX: Metal seems to handle just like 3D txd, so why doesn't it work?
       * TODO: Stop using this lowering
       */
      .lower_txd_cube_map = true,
   };

   NIR_PASS(progress, s, nir_lower_tex, &lower_tex_options);
   return progress;
}

static bool
lower_lod_clamp(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   int min_idx = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   if (min_idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   if (tex->op == nir_texop_txl) {
      /* Explicit LOD has no separate hardware minimum operand. Clamp before
       * legalizing to fp16, at the precision of the requested LOD. */
      nir_def *min_lod = nir_steal_tex_src(tex, nir_tex_src_min_lod);
      int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      assert(lod_idx >= 0);
      nir_def *lod = tex->src[lod_idx].src.ssa;
      nir_src_rewrite(&tex->src[lod_idx].src,
                      nir_fmax(b, lod, nir_f2fN(b, min_lod, lod->bit_size)));
      return true;
   }

   /* txb is packed below; txd keeps its gradient + minimum operand. */
   return false;
}

static bool
saturate_hw_txf_srcs(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP)
      return false;

   nir_tex_src_type src_type;
   if (tex->op == nir_texop_txf)
      src_type = nir_tex_src_lod;
   else if (tex->op == nir_texop_txf_ms)
      src_type = nir_tex_src_ms_index;
   else
      return false;

   int idx = nir_tex_instr_src_index(tex, src_type);
   if (idx < 0)
      return false;

   nir_def *val = tex->src[idx].src.ssa;
   if (val->bit_size <= 16)
      return false;

   /* Unsigned saturation to 0xFFFF preserves every valid level/sample while
    * preventing 32-bit values such as 65536 or -65536 from truncating to 0
    * when nir_legalize_16bit_sampler_srcs() narrows the source to 16 bits. */
   b->cursor = nir_before_instr(&tex->instr);
   nir_src_rewrite(&tex->src[idx].src,
                   nir_umin(b, val, nir_imm_intN_t(b, 65535, val->bit_size)));
   return true;
}

static bool
lower_texture(nir_shader *s, unsigned generation, bool buffer_metadata,
              bool robust_formatless_images, bool robust_texel_fetch,
              uint64_t sparse_ro_offset, bool sparse_buffer_residency,
              agx_buffer_texture_metadata_cb buffer_metadata_cb)
{
   bool progress = false;
   const bool g18g = generation & AGX_NIR_TEXTURE_GENERATION_G18G;
   const bool image_sideband = generation & AGX_NIR_TEXTURE_IMAGE_SIDEBAND;
   const bool hw_txf_bounds = generation & AGX_NIR_TEXTURE_HW_TXF_BOUNDS;
   const bool level_zero_extent = generation & AGX_NIR_TEXTURE_LEVEL_ZERO_EXTENT;
   generation &= ~(AGX_NIR_TEXTURE_LEVEL_ZERO_EXTENT |
                   AGX_NIR_TEXTURE_HW_TXF_BOUNDS |
                   AGX_NIR_TEXTURE_GENERATION_G18G |
                   AGX_NIR_TEXTURE_IMAGE_SIDEBAND);
   const bool gen9 = generation == 9;
   const struct agx_texture_lower_options options = {
      .gen9 = gen9,
      .gen10 = generation == 10,
      .g18g = g18g && generation == 10,
      .buffer_metadata = buffer_metadata,
      .image_sideband = image_sideband,
      .level_zero_extent = level_zero_extent,
      .buffer_metadata_cb = buffer_metadata_cb,
      .robust_formatless_images = robust_formatless_images,
      .robust_texel_fetch = robust_texel_fetch,
      .hw_txf_bounds = hw_txf_bounds,
      .sparse_ro_offset = sparse_ro_offset,
      .sparse_buffer_residency = sparse_buffer_residency,
   };

   nir_tex_src_type_constraints tex_constraints = {
      [nir_tex_src_lod] = {true, 16},
      [nir_tex_src_bias] = {true, 16},
      [nir_tex_src_ms_index] = {true, 16},
      [nir_tex_src_min_lod] = {true, 16},
      [nir_tex_src_texture_offset] = {true, 16},
      [nir_tex_src_sampler_offset] = {true, 16},
   };

   /* Insert fences before lowering image atomics, since image atomics need
    * different fencing than other image operations.
    */
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, fence_image,
            nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_lower_image_atomics_to_global, NULL, NULL);

   NIR_PASS(progress, s, nir_shader_intrinsics_pass, legalize_image_lod,
            nir_metadata_control_flow, NULL);
   NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_images,
            nir_metadata_control_flow, (void *)&options);
   NIR_PASS(progress, s, nir_shader_tex_pass, lower_lod_clamp,
            nir_metadata_control_flow, NULL);
   if (hw_txf_bounds) {
      NIR_PASS(progress, s, nir_shader_tex_pass, saturate_hw_txf_srcs,
               nir_metadata_control_flow, NULL);
   }
   NIR_PASS(progress, s, nir_legalize_16bit_sampler_srcs, tex_constraints);

   /* Fold constants after nir_legalize_16bit_sampler_srcs so we can detect 0 in
    * lower_regular_texture. This is required for correctness.
    */
   NIR_PASS(progress, s, nir_opt_constant_folding);

   /* A sparse texel-buffer fetch takes its residency from software, so de-
    * sparse it before lower_buffer_texture() turns it into a 2D fetch. */
   if (buffer_metadata) {
      NIR_PASS(progress, s, nir_shader_instructions_pass,
               lower_sparse_buffer_texture, nir_metadata_control_flow,
               (void *)&options);
   }

   /* Lower texture sources after legalizing types (as the lowering depends on
    * 16-bit multisample indices) but before lowering queries (as the lowering
    * generates txs for array textures).
    */
   NIR_PASS(progress, s, lower_buffer_format_groups, &options);
   NIR_PASS(progress, s, nir_shader_instructions_pass, lower_regular_texture,
            nir_metadata_none, (void *)&options);
   NIR_PASS(progress, s, nir_shader_instructions_pass, lower_tex_crawl,
            nir_metadata_control_flow, (void *)&options);

   if (gen9 || generation == 10) {
      /* AGX3's texture2d::read consumes a full uint LOD through the integer
       * PACK_SMPIR operand. Keep the shared 16-bit lowering above for its
       * robustness arithmetic, then restore the hardware ABI on the final
       * texel-fetch instruction. */
      nir_foreach_function_impl(impl, s) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_tex)
                  continue;
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               if (tex->op != nir_texop_txf)
                  continue;
               const int lod =
                  nir_tex_instr_src_index(tex, nir_tex_src_lod);
               if (lod < 0 || nir_src_bit_size(tex->src[lod].src) == 32)
                  continue;
               assert(nir_src_bit_size(tex->src[lod].src) == 16);
               nir_builder b = nir_builder_at(nir_before_instr(instr));
               nir_src_rewrite(&tex->src[lod].src,
                               nir_u2u32(&b, tex->src[lod].src.ssa));
               progress = true;
            }
         }
      }
   }

   return progress;
}

bool
agx_nir_lower_texture_for_generation(nir_shader *s, unsigned generation)
{
   return lower_texture(s, generation, false, false, true, 0, false, NULL);
}

bool
agx_nir_lower_texture_with_buffer_metadata(nir_shader *s, unsigned generation,
                                           bool robust_formatless_images,
                                           uint64_t sparse_ro_offset,
                                           bool sparse_buffer_residency)
{
   /* hk's image robustness covers both formatless loads and texel fetches. */
   return lower_texture(s, generation, true, robust_formatless_images,
                        robust_formatless_images, sparse_ro_offset,
                        sparse_buffer_residency, NULL);
}

bool
agx_nir_lower_texture_with_buffer_metadata_cb(
   nir_shader *s, unsigned generation,
   agx_buffer_texture_metadata_cb buffer_metadata, bool robust_texel_fetch)
{
   return lower_texture(s, generation, true, false, robust_texel_fetch, 0,
                        false, buffer_metadata);
}

bool
agx_nir_lower_texture(nir_shader *s)
{
   return agx_nir_lower_texture_for_generation(s, 0);
}

static bool
lower_multisampled_store(nir_builder *b, nir_intrinsic_instr *intr,
                         void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic != nir_intrinsic_bindless_image_store)
      return false;

   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_MS)
      return false;

   nir_def *index_px;
   const unsigned generation = *(unsigned *)data;
   /* G18G: native eMRT tables carry software geometry outside the hardware
    * PBE. This path is selected by the epilog's explicit private-table ABI. */
   if (generation == 9 ||
       generation == (10 | AGX_NIR_TEXTURE_GENERATION_G18G)) {
      nir_def *pbe = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
      index_px = libagx_emrt_ms_index_gen9(b,
         nir_iadd_imm(b, pbe, AGX_G17P_IMAGE_ATOMIC_SW_FROM_PBE),
         nir_iadd_imm(b, pbe, AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET - 32),
         intr->src[1].ssa, nir_u2u32(b, intr->src[2].ssa),
         nir_imm_int(b, util_format_get_blocksize(nir_intrinsic_format(intr))));
   } else {
      index_px = nir_u2u32(b, image_texel_address(b, intr, true, false, false, false,
                                                     false));
   }
   nir_def *coord2d = coords_for_buffer_texture(b, index_px);

   nir_src_rewrite(&intr->src[1], nir_pad_vector(b, coord2d, 4));
   nir_src_rewrite(&intr->src[2], nir_imm_int(b, 0));
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
   nir_intrinsic_set_image_array(intr, false);
   return true;
}

bool
agx_nir_lower_multisampled_image_store(nir_shader *s)
{
   return agx_nir_lower_multisampled_image_store_for_generation(s, 0);
}

bool
agx_nir_lower_multisampled_image_store_for_generation(nir_shader *s,
                                                     unsigned generation)
{
   return nir_shader_intrinsics_pass(s, lower_multisampled_store,
                                     nir_metadata_control_flow, &generation);
}

/*
 * Given a non-bindless instruction, return whether agx_nir_lower_texture will
 * lower it to something involving a descriptor crawl. This requires the driver
 * to lower the instruction to bindless before calling agx_nir_lower_texture.
 * The implementation just enumerates the cases handled in this file.
 */
bool
agx_nir_needs_texture_crawl(nir_instr *instr)
{
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      /* Queries, atomics always become a crawl */
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_samples:
      case nir_intrinsic_image_deref_samples:
      case nir_intrinsic_image_atomic:
      case nir_intrinsic_image_deref_atomic:
      case nir_intrinsic_image_atomic_swap:
      case nir_intrinsic_image_deref_atomic_swap:
         return true;

      /* Multisampled stores need a crawl, others do not */
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_deref_store:
         return nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS;

      /* Array loads need a crawl, other load do not */
      case nir_intrinsic_image_load:
         return nir_intrinsic_image_array(intr) ||
                nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_CUBE;

      default:
         return false;
      }
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      /* Array textures get clamped to their size via txs */
      if (tex->is_array && !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP))
         return true;

      switch (tex->op) {
      /* Queries always become a crawl */
      case nir_texop_txs:
      case nir_texop_texture_samples:
      case nir_texop_query_levels:
         return true;

      /* Buffer textures need their format read and txf needs its LOD/layer
       * clamped.  Buffer textures are only read through txf.
       */
      case nir_texop_txf:
      case nir_texop_txf_ms:
         return has_nonzero_lod(tex) || tex->is_array ||
                tex->sampler_dim == GLSL_SAMPLER_DIM_BUF;

      default:
         return false;
      }
   }

   return false;
}

/* Walk uses once, including loop-carried phis. A depth limit can miss a
 * sampled value after a long ALU chain; recursive traversal can also revisit
 * the same phi indefinitely. */
static bool
agx_g17p_reaches_derivative(nir_def *def)
{
   struct set *visited = _mesa_pointer_set_create(NULL);
   struct util_dynarray pending;
   util_dynarray_init(&pending, NULL);
   util_dynarray_append(&pending, def);
   bool found = false;

   while (util_dynarray_num_elements(&pending, nir_def *) && !found) {
      nir_def *current = util_dynarray_pop(&pending, nir_def *);
      if (_mesa_set_search(visited, current))
         continue;
      _mesa_set_add(visited, current);

      nir_foreach_use(use, current) {
         nir_instr *instr = nir_src_use_instr(use);
         if (instr->type == nir_instr_type_intrinsic) {
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_ddx:
            case nir_intrinsic_ddx_fine:
            case nir_intrinsic_ddx_coarse:
            case nir_intrinsic_ddy:
            case nir_intrinsic_ddy_fine:
            case nir_intrinsic_ddy_coarse:
               found = true;
               break;
            default:
               break;
            }
         } else if (instr->type == nir_instr_type_alu) {
            util_dynarray_append(&pending, &nir_instr_as_alu(instr)->def);
         } else if (instr->type == nir_instr_type_phi) {
            util_dynarray_append(&pending, &nir_instr_as_phi(instr)->def);
         }
      }
   }

   util_dynarray_fini(&pending);
   _mesa_set_destroy(visited, NULL);
   return found;
}

/* G17P's driver-profile implicit sample returns zero in helper lanes even
 * under the native helper-capable fragment protocol.  That is unobservable
 * unless the sampled value itself feeds a later derivative.  For that exact
 * shape, spell the implicit LOD as the equivalent explicit gradients: the
 * witnessed gradient sampler executes in helper lanes and preserves the quad
 * values required by fwidth(texture(...)). */
static bool
agx_g17p_explicit_helper_gradient(nir_builder *b, nir_tex_instr *tex,
                                  UNUSED void *data)
{
   if (tex->op != nir_texop_tex ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_array ||
       tex->is_shadow || tex->is_sparse ||
       !agx_g17p_reaches_derivative(&tex->def))
      return false;

   const int coord_idx =
      nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord_idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   nir_tex_instr_add_src(tex, nir_tex_src_ddx, nir_ddx(b, coord));
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, nir_ddy(b, coord));
   tex->op = nir_texop_txd;
   return true;
}

bool
agx_nir_lower_g17p_helper_samples(nir_shader *shader)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);
   return nir_shader_tex_pass(shader, agx_g17p_explicit_helper_gradient,
                              nir_metadata_control_flow, NULL);
}

bool
agx_nir_lower_texture_early_with_2d_robustness(nir_shader *s,
                                               bool support_lod_bias,
                                               bool lower_2d_robustness)
{
   return lower_texture_early(s, support_lod_bias, lower_2d_robustness, false);
}
