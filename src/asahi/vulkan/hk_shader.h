/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir.h"
#include "poly/nir/poly_nir.h"
#include "util/macros.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "agx_usc.h"
#include "agx_uvs.h"
#include "agx_vdm_state_update_core.h"

#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_private.h"
#include "hk_nir_lower_mesh.h"

#include "nir_xfb_info.h"
#include "nir.h"
#include "shader_enums.h"
#include "vk_pipeline_cache.h"

#include "nir_defines.h"

#include "vk_shader.h"

struct hk_physical_device;
struct hk_pipeline_compilation_ctx;
struct vk_descriptor_set_layout;
struct vk_graphics_pipeline_state;
struct vk_pipeline_cache;
struct vk_pipeline_layout;
struct vk_pipeline_robustness_state;
struct vk_shader_module;

struct hk_tess_info {
   enum tess_primitive_mode mode : 8;
   enum gl_tess_spacing spacing  : 8;
   bool points;
   bool ccw;
};
static_assert(sizeof(struct hk_tess_info) == 4, "packed");

static struct hk_tess_info
hk_tess_info_merge(struct hk_tess_info a, struct hk_tess_info b)
{
   static_assert(TESS_PRIMITIVE_UNSPECIFIED == 0, "zero state");
   static_assert(TESS_SPACING_UNSPECIFIED == 0, "zero state");

   /* Just merge by OR'ing the raw bits */
   uint32_t x, y;
   memcpy(&x, &a, sizeof(x));
   memcpy(&y, &b, sizeof(y));

   x |= y;

   struct hk_tess_info out;
   memcpy(&out, &x, sizeof(out));
   return out;
}

struct hk_shader_info {
   union {
      struct {
         uint32_t attribs_read;
         BITSET_DECLARE(attrib_components_read, AGX_MAX_ATTRIBS * 4);
         bool use_prolog;
         /* G17P: the shader places the three vertices of a draw that starts
          * at vertex 0 and instance 0 by the vertex index alone, on a
          * triangle covering the whole clip-space square in front of the
          * viewer (hk_vs_covers_clip_square()). */
         bool covers_clip_square;
         uint8_t _pad[6];
      } vs;

      struct {
         struct agx_interp_info interp;
         struct agx_fs_epilog_link_info epilog_key;

         bool reads_sample_mask;
         bool post_depth_coverage;
         bool uses_sample_shading;
         bool early_fragment_tests;
         bool writes_memory;

         /* Rasterization samples the G17P coverage updates are compiled for */
         uint8_t g17p_samples;

         /* AGX_G17P_UNIT_RANGE_BLEND: every colour output is provably in
          * [0, 1] (computed only with the flag set). */
         bool g17p_colour_unit_range;

         /* AGX3: the main part's depth/stencil test selector was folded to
          * "this part tests" (hk_fs_key::static_tests); its epilog must not
          * discard. */
         bool static_tests;
         /* G17P: the shader may leave pixels unwritten: it discards, demotes
          * or writes the sample mask. */
         bool may_kill;
         bool m5_inline_sample_shading;
         /* G17P: the main part applies alpha-to-coverage in the coverage
          * update that tests (agx_fs_epilog_key::a2c_in_main). */
         bool a2c_in_main;
         uint8_t _pad[1];
      } fs;

      struct {
         uint64_t tcs_per_vertex_outputs;
         uint32_t tcs_output_stride;
         uint8_t tcs_output_patch_size;
         uint8_t tcs_nr_patch_outputs;

         struct hk_tess_info info;
      } tess;

      struct poly_gs_info gs;



      /* Used to initialize the union for other stages */
      uint8_t _pad[32];
   };

   struct hk_mesh_info mesh;
   struct agx_unlinked_uvs_layout uvs;

   /* Transform feedback buffer strides */
   uint8_t xfb_stride[MAX_XFB_BUFFERS];

   mesa_shader_stage stage : 8;
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;
   uint8_t set_count;

   /* Positive proof recorded before inserting the default Layer output.
    * Only original vertex hardware variants qualify.  False is unknown.
    * Serialized with shader binary version 26 or newer.
    */
   bool g17p_default_layer_zero;

   /* External API memory writes (stores, atomics, transform feedback). */
   bool writes_external_memory;

   /* The shader samples, loads, stores or queries texel contents of an image
    * (any texture instruction or image access, bindless included). On G17P
    * a render whose pre-rasterization shaders read no image does not order
    * its vertex half after image-only barriers (hk_sync.h, HK_SYNC_GFX_IMG).
    */
   bool reads_images;

   /* The bounded G17P graphics/compute path lowers one Gen9 combined
    * image/sampler to native texture/sampler state 0. Command recording joins
    * the bound descriptor set and native sampler heap into those CProg slots.
    */
   bool agx3_combined_sampler_valid;
   bool agx3_uses_root;
   bool agx3_gen1_fragment_rq;
   /* HK_ROOT_READS_* classes of root bytes the shader may read. */
   uint8_t agx3_root_reads;
   /* Actual uniform address selected after stage and resource lowering. */
   unsigned agx3_root_uniform;
   bool agx3_txf_texture_valid;
   bool agx3_storage_image_valid;
   bool agx3_sampled_storage_valid;
   bool agx3_indexed_sampler_valid;
   bool agx3_indexed_storage_valid;
   struct agx_g17p_combined_sampler_binding agx3_combined_sampler;
   struct agx_g17p_txf_binding agx3_txf_texture;
   struct agx_g17p_storage_image_binding agx3_storage_image;
   struct agx_g17p_sampled_storage_binding agx3_sampled_storage;
   struct agx_g17p_indexed_storage_binding agx3_indexed_storage;

   /* XXX: is there a less goofy way to do this? I really don't want dynamic
    * allocation here.
    *
    * nir_gather_xfb_info() emits at most one output per component of a
    * captured slot, and vertex-pipeline outputs occupy the slots below
    * VARYING_SLOT_VAR31 + 1. Sixty-four entries overflowed into the next
    * variant (a CTS vertex shader captures 94 outputs).
    */
   nir_xfb_info xfb_info;
   nir_xfb_output_info xfb_outputs[4 * (VARYING_SLOT_VAR31 + 1)];
};

/*
 * Hash table keys for fast-linked shader variants. These contain the entire
 * prolog/epilog key so we only do 1 hash table lookup instead of 2 in the
 * general case where the linked shader is already ready.
 */
struct hk_fast_link_key_vs {
   struct agx_vs_prolog_key prolog;
};

struct hk_fast_link_key_fs {
   unsigned nr_samples_shaded;
   struct agx_fs_prolog_key prolog;
   struct agx_fs_epilog_key epilog;
};

struct hk_shader {
   struct agx_shader_part b;

   struct hk_shader_info info;
   /* G17P-only root prefix; serialized separately in its version-35 payload.
    * Keep hk_shader_info and version-34 payloads unchanged on other chips. */
   uint32_t root_descriptor_size;
   /* G17P mesh: 32-bit words each varying slot takes in the per-primitive
    * record (agx_gen1_mesh_record_layout()), serialized like the root prefix.
    * A fragment shader compiled without this mesh shader reads its
    * per-primitive inputs through hk_mesh_fs_twin(). */
   uint8_t g17p_mesh_primitive_comps[64];
   struct agx_fragment_face_2_packed frag_face;
   struct agx_counts_packed counts;

   /* Native AGX3 VertexProgramVariant state.  G17P hardware vertex shaders
    * with position and up to two ordinary varying slots populate the exact
    * no-preamble fields; other AGX3 layouts remain zero. */
   struct agx_vdm_state_update_program_info agx3_vdm_state_update;

   const void *code_ptr;
   uint32_t code_size;

   const void *data_ptr;
   uint32_t data_size;
   /* Process-unique identity of the immutable payload, shared by cache clones. */
   uint64_t constant_data_id;

   /* BO for any uploaded shader part */
   /* Code heap chunk holding the binary, at bo_offset. */
   struct agx_bo *bo;
   uint32_t bo_offset;

   /* Cache of fast linked variants */
   struct {
      simple_mtx_t lock;
      struct hash_table *ht;
      bool lazy_ht;
   } linked;

   /* The variant was compiled (hk_compile_nir) or deserialized from a
    * compiled variant.  False only for a variant that was never compiled,
    * e.g. the software vertex variant of a shader without a tessellation or
    * geometry successor.  A compiled variant may have zero bytes of code: a
    * G17P fragment shader with no work (an empty main) is all prolog and
    * epilog, so code_size == 0 does not mean "never compiled".
    */
   bool compiled;

   /* If there's only a single possibly linked variant, direct pointer. TODO:
    * Union with the cache to save some space?
    */
   struct hk_linked_shader *only_linked;

   /* Address to the uploaded preamble section. Preambles are uploaded
    * separately from fast-linked main shaders.
    */
   uint64_t preamble_addr;

   /* Address of the start of the shader data section */
   uint64_t data_addr;

   /* The fragment main's NIR (serialized just before its backend compile)
    * and that compile's key, so fast-link can build one program with the
    * draw's colour epilog inlined. NULL when the shape is not eligible. */
   void *g17p_mono_nir;
   size_t g17p_mono_nir_size;
   struct agx_shader_key g17p_mono_key;
};

static inline uint64_t
hk_shader_code_va(const struct hk_shader *s)
{
   return s->bo->va->addr + s->bo_offset;
}

enum hk_vs_variant {
   /* Hardware vertex shader, when next stage is fragment */
   HK_VS_VARIANT_HW,

   /* Hardware compute shader, when next is geometry/tessellation */
   HK_VS_VARIANT_SW,

   HK_VS_VARIANTS,
};

enum hk_gs_variant {
   /* Hardware vertex shader used for rasterization */
   HK_GS_VARIANT_RAST,

   /* Main compute shader */
   HK_GS_VARIANT_MAIN,

   /* Count compute shader */
   HK_GS_VARIANT_COUNT,

   /* Pre-GS compute shader */
   HK_GS_VARIANT_PRE,

   HK_GS_VARIANTS,
};

/* clang-format off */
static const char *hk_gs_variant_name[] = {
   [HK_GS_VARIANT_RAST] = "Rasterization",
   [HK_GS_VARIANT_MAIN] = "Main",
   [HK_GS_VARIANT_COUNT] = "Count",
   [HK_GS_VARIANT_PRE] = "Pre-GS",
};
/* clang-format on */

enum hk_mesh_variant {
   /* Main compute kernel writing vertex/primitive/index records */
   HK_MESH_VARIANT_MAIN,

   /* Hardware vertex shader rasterizing those records */
   HK_MESH_VARIANT_RAST,

   /* Native object predicate/count kernel; selected only when required. */
   HK_MESH_VARIANT_PREDICATED_OBJECT,

   HK_MESH_VARIANTS,
};

/* clang-format off */
static const char *hk_mesh_variant_name[] = {
   [HK_MESH_VARIANT_MAIN] = "Main",
   [HK_MESH_VARIANT_RAST] = "Passthrough object",
   [HK_MESH_VARIANT_PREDICATED_OBJECT] = "Predicated object",
};
/* clang-format on */

static inline unsigned
hk_num_variants(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_EVAL:
      return HK_VS_VARIANTS;

   case MESA_SHADER_GEOMETRY:
      return HK_GS_VARIANTS;

   case MESA_SHADER_MESH:
      return HK_MESH_VARIANTS;

   default:
      return 1;
   }
}

/*
 * A G17P hardware vertex program whose UVS layout does not start at component
 * zero with a four-component position group. The AGX3 packer admits a dense
 * terminal output run only from five outputs up, a floor every position-
 * writing shader clears through its four position components alone, so every
 * shape below needs the sparse component-addressed staging protocol instead.
 *
 * `info` is the hardware shader being compiled and `sw_stage` the API stage it
 * serves; `xfb_outputs` says whether the pipeline records transform feedback.
 */
static inline bool
hk_g17p_sparse_vertex_outputs(const struct shader_info *info,
                              mesa_shader_stage sw_stage, bool xfb_outputs)
{
   if (info->stage != MESA_SHADER_VERTEX)
      return false;

   return info->has_transform_feedback_varyings || xfb_outputs ||
          sw_stage == MESA_SHADER_GEOMETRY ||
          /* A raster-discard TES may write only point size, leaving the
           * four position components absent just like an XFB raster VS. */
          sw_stage == MESA_SHADER_TESS_EVAL ||
          info->next_stage == MESA_SHADER_GEOMETRY ||
          /* The hardware variant of a control-point vertex shader is never
           * bound -- the software variant runs in compute -- but it is still
           * compiled, and its UVS layout is as sparse as the geometry case
           * above. Without this the AGX3 packer refuses it at st_vary #4 and
           * takes every G17P tessellation pipeline with it. */
          info->next_stage == MESA_SHADER_TESS_CTRL ||
          /* A vertex shader may legally write no gl_Position at all. The
           * four-entry UVS position group stays reserved, so its user
           * varyings are a dense run based at AGX3_UVS_POSITION_ENTRIES --
           * the same shape as the XFB and geometry cases -- but there are
           * fewer than five of them, so the dense-run admission refuses the
           * store at st_vary #4. That is the whole
           * dEQP-VK.pipeline.*.no_position.basic.*.v0* family. */
          !(info->outputs_written & VARYING_BIT_POS);
}

/*
 * An hk_api shader maps 1:1 to a VkShader object. An hk_api_shader may contain
 * multiple hardware hk_shader's, built at shader compile time. This complexity
 * implements the legacy geometry pipeline and native sample-count variants.
 */
struct hk_api_shader {
   struct vk_shader vk;

   /* Non-NULL only for a fresh API wrapper borrowing immutable deserialized
    * code/data from the device's bounded binary cache. */
   struct hk_shader_binary_cache_entry *binary_cache_entry;

   /* Is this an internal passthrough geometry shader? */
   bool is_passthrough;

   /* Native sample-count variants: three slots on G17P, four on M5. */
   uint8_t g17p_sample_variants;

   /* The one-sample binary compiles with the shader; the multisample ones
    * compile from this source on first use (hk_fs_build_sample_variant). */
   struct hk_fs_sample_source *sample_source;

   /* Non-NULL for a shader compiled without custom-border emulation because
    * the device had created no sampler that needs it: what its emulating
    * twin compiles from, and the twin once built (hk_border_twin). */
   struct hk_border_source *border_source;

   /* Compiled with a custom-border emulation that a device without a sampler
    * needing it would have left out (serialized; such a binary is refused
    * there, so a cache never hands it out unneeded). */
   bool border_emulating;

   /* Compiled with agx_nir_lower_clip_m1_1 where it rasterizes; a mesh shader
    * sets it when its device enabled depthClipControl (serialized: a binary
    * without it is refused by a device that enabled the feature). */
   bool clip_m1_1;

   /* G17P: non-NULL for a fragment shader compiled without its mesh shader
    * that reads per-primitive inputs, gl_PrimitiveID, gl_Layer or
    * gl_ViewportIndex. A mesh shader stores those in per-primitive records
    * whose layout it alone knows, so mesh draws use a twin compiled for that
    * layout from this source (same inputs as a border source); mesh_twins
    * holds the twins built so far, under mesh_source->lock. */
   struct hk_border_source *mesh_source;
   struct hk_mesh_fs_twin *mesh_twins;

   /* Non-NULL for a fragment shader whose depth/stencil feedback loop is only
    * known at draw time: what its zs_self_dep twin compiles from. */
   struct hk_border_source *fbl_source;

   struct hk_shader variants[];
};

/* The custom-border-emulating twin of a shader with a border source, built
 * on first use. Thread-safe; NULL if it cannot be built. */
struct hk_api_shader *hk_border_twin(struct hk_device *dev,
                                     struct hk_api_shader *obj);

/* The twin of a fragment shader with a feedback-loop source, compiled for a
 * depth/stencil feedback loop and built on first use. Thread-safe; NULL if it
 * cannot be built. */
struct hk_api_shader *hk_fbl_twin(struct hk_device *dev,
                                  struct hk_api_shader *obj);

/* The twin of a fragment shader with a mesh source that reads per-primitive
 * inputs from the records of this G17P mesh variant, built on first use.
 * Thread-safe; *out is NULL if it cannot be built. */
VkResult hk_mesh_fs_twin(struct hk_device *dev, struct hk_api_shader *obj,
                         const struct hk_shader *mesh,
                         struct hk_api_shader **out);

/* Make the `samples` variant of a dynamic-sample-count G17P fragment shader
 * available to hk_fs_variant(). Thread-safe; a no-op for other shaders. */
VkResult hk_fs_build_sample_variant(struct hk_device *dev,
                                    struct hk_api_shader *obj,
                                    unsigned samples);

void hk_shader_binary_cache_finish(struct hk_device *dev);
bool hk_shader_ensure_link_ht_locked(struct hk_shader *shader);

static inline unsigned
hk_api_num_variants(const struct hk_api_shader *obj)
{
   return obj->g17p_sample_variants ? obj->g17p_sample_variants
                                    : hk_num_variants(obj->vk.stage);
}

static inline size_t
hk_api_shader_size(mesa_shader_stage stage, unsigned samples)
{
   unsigned count = samples ? samples : hk_num_variants(stage);
   return sizeof(struct hk_api_shader) + count * sizeof(struct hk_shader);
}

#define hk_foreach_variant(api_shader, var)                                    \
   for (struct hk_shader *var = api_shader->variants;                          \
        var < api_shader->variants + hk_api_num_variants(api_shader);          \
        ++var)

static const char *
hk_variant_name(struct hk_api_shader *obj, struct hk_shader *variant)
{
   unsigned i = variant - obj->variants;
   assert(i < hk_api_num_variants(obj));

   if (obj->g17p_sample_variants) {
      static const char *names[] = {"1 sample", "2 samples", "4 samples", "8 samples"};
      return names[i];
   } else if (hk_api_num_variants(obj) == 1) {
      return NULL;
   } else if (obj->vk.stage == MESA_SHADER_GEOMETRY) {
      assert(i < ARRAY_SIZE(hk_gs_variant_name));
      return hk_gs_variant_name[i];
   } else if (obj->vk.stage == MESA_SHADER_MESH) {
      assert(i < ARRAY_SIZE(hk_mesh_variant_name));
      return hk_mesh_variant_name[i];
   } else {
      assert(i < 2);
      return i == HK_VS_VARIANT_SW ? "Software" : "Hardware";
   }
}

static struct hk_shader *
hk_only_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   assert(hk_api_num_variants(obj) == 1);
   return &obj->variants[0];
}

static struct hk_shader *
hk_any_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   return &obj->variants[0];
}

/* Link a complete pipeline's fragment shader for the state the pipeline
 * fixes, so its first draw finds the linked program. */
void hk_prelink_pipeline_fragment(struct hk_device *dev,
                                  const struct hk_shader *hw_vs,
                                  struct hk_api_shader *fs,
                                  const struct vk_graphics_pipeline_state *state);
void hk_prelink_object_fragment(struct hk_device *dev,
                                struct hk_api_shader *fs);
void hk_prelink_vertex(struct hk_device *dev, struct hk_api_shader *vs,
                       bool hw_program, bool sw_program);

static inline struct hk_shader *
hk_fs_variant(struct hk_api_shader *obj, unsigned samples)
{
   if (!obj)
      return NULL;

   assert(obj->vk.stage == MESA_SHADER_FRAGMENT);
   if (obj->g17p_sample_variants)
      assert(samples == 0 || samples == 1 || samples == 2 || samples == 4 ||
             (samples == 8 && obj->g17p_sample_variants == 4));
   if (!obj->g17p_sample_variants)
      return hk_only_variant(obj);

   return &obj->variants[samples > 1 ? util_logbase2(samples) : 0];
}

static struct hk_shader *
hk_main_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_MAIN];
}

static struct hk_shader *
hk_count_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_COUNT];
}

static struct hk_shader *
hk_pre_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_PRE];
}

static struct hk_shader *
hk_main_mesh_variant(struct hk_api_shader *obj)
{
   assert(obj->vk.stage == MESA_SHADER_MESH);
   return &obj->variants[HK_MESH_VARIANT_MAIN];
}

static struct hk_shader *
hk_rast_mesh_variant(struct hk_api_shader *obj)
{
   assert(obj->vk.stage == MESA_SHADER_MESH);
   return &obj->variants[HK_MESH_VARIANT_RAST];
}

#define HK_MAX_LINKED_USC_SIZE                                                 \
   (AGX_USC_PRESHADER_LENGTH + AGX_USC_FRAGMENT_PROPERTIES_LENGTH +            \
    AGX_USC_REGISTERS_LENGTH + AGX_USC_SHADER_LENGTH + AGX_USC_SHARED_LENGTH + \
    AGX_USC_SAMPLER_LENGTH + (AGX_USC_UNIFORM_LENGTH * 9))

struct hk_linked_shader {
   struct agx_linked_shader b;

   /* A specialized fragment executable may own a different constant program
    * and uniform layout. Its launch source lives as long as this executable. */
   struct hk_shader *variant_source;

   /* Immutable G17P compute-program templates keyed by uniform-range shape.
    * Entries are published lock-free while command buffers record in
    * parallel and are released with this linked executable. */
   void *g17p_compute_templates;

   /* M5 compute loaders retain their exact native load schedule. */
   void *m5_compute_templates;

   /* G17P stage programs of this executable, one per uniform-range layout,
    * patched with each draw's range addresses (struct hk_g17p_gfx_loader). */
   void *g17p_gfx_loaders;

   /* True if the VS prolog uses software indexing, either for geom/tess or
    * adjacency primitives.
    */
   bool sw_indexing;

   /* True when sw_indexing is set purely because the topology carries
    * adjacency, i.e. the bound program is still the hardware vertex shader
    * and only fetches its own indices. The geometry/tessellation case, where
    * the software vertex shader runs as a compute kernel, is a different
    * shape and keeps sw_indexing with this clear. */
   bool sw_indexing_hw_vs;

   /* Absolute uniform-half base selected by the fragment epilog key for its
    * dynamic blend constants. Zero means the standard half4 ABI. */
   uint16_t g17p_blend_constant_base;
   uint16_t g17p_emrt_heap_base;

   /* The pipeline-statistics fragment prolog loads its counter address from
    * the descriptor root even when the application fragment shader does not. */
   bool g17p_prolog_uses_root;

   /* An inline-blend fragment program that forms its own constant program
    * (hk_g17p_inline_epilog_part()) has its own uniform layout: this shader
    * describes it, and the draw takes the program's launch (uniform ranges,
    * constant program, scratch) from it instead of the API shader. Owned by
    * this linked shader; NULL when the API shader's layout applies. */
   struct hk_shader *g17p_layout;

   /* Distinct from hk_shader::counts due to addition of cf_binding_count, which
    * is delayed since it depends on cull distance.
    */
   struct agx_fragment_shader_word_0_packed fs_counts;

   /* Baked USC words to bind this linked shader */
   struct {
      uint8_t data[HK_MAX_LINKED_USC_SIZE];
      size_t size;
   } usc;
};

/* Executable resource requirements may differ from the API shader's after
 * specialization. Use this for both loader and surrounding launch records. */
static inline struct hk_shader *
hk_linked_shader_source(struct hk_shader *source,
                        const struct hk_linked_shader *linked)
{
   return linked && linked->variant_source ? linked->variant_source : source;
}

/* `main_part`, when non-NULL, replaces main->b as the linked main part
 * (the inline-blend monolithic fragment program); main still supplies the
 * API shader's uniform and resource layout. */
struct hk_linked_shader *hk_fast_link_part(struct hk_device *dev, bool fragment,
                                           struct hk_shader *main,
                                           struct agx_shader_part *main_part,
                                           struct agx_shader_part *prolog,
                                           struct agx_shader_part *epilog,
                                           unsigned nr_samples_shaded);
/* A copy of `api` whose code, uniform layout and constant program are
 * `part`'s (an inline-blend program with its own constant program); takes
 * ownership of part->binary. */
struct hk_shader *hk_g17p_layout_shader_create(struct hk_device *dev,
                                               const struct hk_shader *api,
                                               struct agx_shader_part *part);

struct hk_linked_shader *hk_fast_link(struct hk_device *dev, bool fragment,
                                      struct hk_shader *main,
                                      struct agx_shader_part *prolog,
                                      struct agx_shader_part *epilog,
                                      unsigned nr_samples_shaded);

struct hk_linked_shader *
hk_fast_link_fragment_variant(struct hk_device *dev,
                              const struct hk_shader *source,
                              const struct agx_shader_part *part,
                              const struct agx_fs_epilog_key *key);

extern const struct vk_device_shader_ops hk_device_shader_ops;

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev);

/* The bounded native CProg shape both graphics stages admit: imported
 * uniform inputs stored in, and exactly filling, the preamble's TPS frame,
 * with a known staging footprint. */
static inline bool
hk_g17p_native_cprog_qualified(const struct agx_shader_info *info)
{
   return info->has_preamble && info->preamble_size &&
          info->agx3_preamble_staging_known && info->nr_preamble_gprs &&
          info->nr_preamble_gprs <= 256 &&
          (info->preamble_scratch_size ==
              info->agx3_preamble_inputs.storage_bytes ||
           agx3_cprog_static_spill_info_valid(info)) &&
          (!!info->agx3_preamble_inputs.count ==
           info->agx3_preamble_tps_inputs);
}

/* A null uniform buffer is based at the zero page, which spans
 * maxUniformBufferRange, so null descriptors need no bounds when uniform
 * buffer robustness is disabled. Null storage buffers still need them to
 * discard stores. */
static inline nir_address_format
hk_ubo_addr_format(VkPipelineRobustnessBufferBehaviorEXT robustness)
{
   return robustness == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT
             ? nir_address_format_64bit_global_32bit_offset
             : nir_address_format_64bit_bounded_global;
}

static inline nir_address_format
hk_buffer_addr_format(VkPipelineRobustnessBufferBehaviorEXT robustness,
                      bool null_descriptor, bool map_metadata)
{
   /* A null buffer descriptor has a zero range.  Preserve that range through
    * explicit-I/O lowering even when ordinary robustness is disabled so null
    * loads return zero and null stores are discarded before touching the
    * read-only zero page. */
   if (null_descriptor)
      return map_metadata ? nir_address_format_64bit_bounded_global_map
                          : nir_address_format_64bit_bounded_global;

   switch (robustness) {
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT:
      return map_metadata ? nir_address_format_64bit_global_32bit_offset_map
                          : nir_address_format_64bit_global_32bit_offset;
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT:
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT:
      return map_metadata ? nir_address_format_64bit_bounded_global_map
                          : nir_address_format_64bit_bounded_global;
   default:
      UNREACHABLE("Invalid robust buffer access behavior");
   }
}

bool hk_lower_uvs_index(nir_shader *s, mesa_shader_stage sw_stage,
                        unsigned nr_vbos, unsigned uniform_base);

bool
hk_nir_lower_descriptors(nir_shader *nir,
                         const struct vk_pipeline_robustness_state *rs,
                         uint32_t set_layout_count,
                         struct vk_descriptor_set_layout *const *set_layouts,
                         bool sparse_buffer_residency,
                         bool sparse_soft_unmapped,
                         uint64_t sparse_ro_offset, bool scaled_sampling,
                         bool sampler_has_lod_bias, bool d3d_nearest_mip,
                         bool coherent_images_in_memory,
                         bool m5_store_window);

VkResult hk_compile_shader(struct hk_device *dev,
                           struct vk_shader_compile_info *info,
                           const struct vk_graphics_pipeline_state *state,
                           const struct vk_features *features,
                           const VkAllocationCallbacks *pAllocator,
                           struct hk_api_shader **shader_out);

void hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev,
                                nir_shader *nir);

void hk_api_shader_destroy(struct vk_device *vk_dev,
                           struct vk_shader *vk_shader,
                           const VkAllocationCallbacks *pAllocator);

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs);

struct hk_api_shader *hk_meta_shader(struct hk_device *dev,
                                     hk_internal_builder_t builder, void *data,
                                     size_t data_size, VkResult *result);
struct spirv_to_nir_options
hk_get_spirv_options(struct vk_physical_device *vk_pdev,
                     UNUSED mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *rs);

/* True when a native TES ray query runs in Poly's compute variant. */
bool hk_tes_ray_queries_in_compute(const struct hk_device *dev);
