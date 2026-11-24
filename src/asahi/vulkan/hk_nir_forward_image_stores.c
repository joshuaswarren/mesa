/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_nir_forward_image_stores.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_format_convert.h"
#include "util/format/u_format.h"

/*
 * G17P image reads do not observe writes issued earlier in the same shader
 * invocation, whereas Vulkan requires intra-invocation read-after-write
 * visibility. Forward each image's latest store to subsequent loads of the same
 * coordinate and sample via function-local variables (converted through the
 * image format) while leaving the store itself intact.
 */

#define MAX_TRACKED_IMAGES 8

struct tracked_image {
   nir_variable *image;
   enum pipe_format format;
   enum glsl_sampler_dim dim;
   bool array;
   nir_alu_type type;
   bool indexed;
   bool loaded, stored, unsupported;

   nir_variable *valid, *coord, *index, *sample, *value;
};

struct forward_state {
   struct tracked_image images[MAX_TRACKED_IMAGES];
   unsigned count;
   bool overflow;
};

static bool
format_round_trips(enum pipe_format format)
{
   if (format == PIPE_FORMAT_NONE)
      return false;

   if (format == PIPE_FORMAT_R11G11B10_FLOAT ||
       format == PIPE_FORMAT_R9G9B9E5_FLOAT)
      return true;

   const struct util_format_description *desc =
      util_format_description(format);
   if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB)
      return false;

   for (unsigned c = 0; c < desc->nr_channels; ++c) {
      if (desc->channel[c].type == UTIL_FORMAT_TYPE_FIXED ||
          desc->channel[c].size > 32 ||
          (desc->channel[c].type == UTIL_FORMAT_TYPE_FLOAT &&
           desc->channel[c].size != 16 && desc->channel[c].size != 32))
         return false;
   }

   return true;
}

/* The image variable and, for an array of images, the index into it. Deeper
 * chains are left alone. */
static nir_variable *
image_access_variable(nir_intrinsic_instr *intr, nir_def **index)
{
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   *index = NULL;

   if (deref->deref_type == nir_deref_type_var)
      return deref->var;

   if (deref->deref_type != nir_deref_type_array)
      return NULL;

   nir_deref_instr *parent = nir_deref_instr_parent(deref);
   if (!parent || parent->deref_type != nir_deref_type_var)
      return NULL;

   *index = deref->arr.index.ssa;
   return parent->var;
}

static struct tracked_image *
track_image(struct forward_state *state, nir_variable *image)
{
   for (unsigned i = 0; i < state->count; ++i) {
      if (state->images[i].image == image)
         return &state->images[i];
   }

   if (state->count == MAX_TRACKED_IMAGES) {
      state->overflow = true;
      return NULL;
   }

   state->images[state->count] = (struct tracked_image){.image = image};
   return &state->images[state->count++];
}

static void
gather_access(struct forward_state *state, nir_intrinsic_instr *intr)
{
   nir_def *index;
   nir_variable *image = image_access_variable(intr, &index);
   if (!image) {
      /* An access this pass cannot name may alias a tracked image. */
      state->overflow = true;
      return;
   }

   struct tracked_image *t = track_image(state, image);
   if (!t)
      return;

   const bool load = intr->intrinsic == nir_intrinsic_image_deref_load;
   const bool store = intr->intrinsic == nir_intrinsic_image_deref_store;
   if (!load && !store) {
      /* Atomics read and write the image memory directly. */
      t->unsupported = true;
      return;
   }

   /* Volatile image reads must reach memory, not a remembered store. */
   if (load && (nir_intrinsic_access(intr) & ACCESS_VOLATILE))
      t->unsupported = true;

   const nir_alu_type type = load ? nir_intrinsic_dest_type(intr)
                                  : nir_intrinsic_src_type(intr);
   const nir_def *value = load ? &intr->def : intr->src[3].ssa;
   if (value->num_components != 4 || value->bit_size != 32 ||
       nir_alu_type_get_type_size(type) != 32 ||
       (index && index->num_components != 1))
      t->unsupported = true;

   if (!t->loaded && !t->stored) {
      t->format = nir_intrinsic_format(intr);
      t->dim = nir_intrinsic_image_dim(intr);
      t->array = nir_intrinsic_image_array(intr);
      t->type = type;
      t->indexed = index != NULL;
   } else if (t->format != nir_intrinsic_format(intr) ||
              t->dim != nir_intrinsic_image_dim(intr) ||
              t->array != nir_intrinsic_image_array(intr) ||
              t->type != type || t->indexed != (index != NULL)) {
      t->unsupported = true;
   }

   if (!format_round_trips(nir_intrinsic_format(intr)))
      t->unsupported = true;

   t->loaded |= load;
   t->stored |= store;
}

static unsigned
coordinate_count(const struct tracked_image *t)
{
   return glsl_get_sampler_dim_coordinate_components(t->dim) +
          (t->array ? 1 : 0);
}

static const struct glsl_type *
value_type(const struct tracked_image *t)
{
   switch (nir_alu_type_get_base_type(t->type)) {
   case nir_type_int:
      return glsl_ivec4_type();
   case nir_type_uint:
      return glsl_uvec4_type();
   default:
      return glsl_vec4_type();
   }
}

static nir_def *
in_bounds(nir_builder *b, const struct tracked_image *t,
          nir_intrinsic_instr *intr, nir_def *coord)
{
   const unsigned n = coordinate_count(t);
   unsigned size_components = n;
   if (t->dim == GLSL_SAMPLER_DIM_CUBE && !t->array)
      size_components -= 1;

   nir_def *deref = intr->src[0].ssa;
   const unsigned lod_src =
      intr->intrinsic == nir_intrinsic_image_deref_load ? 3 : 4;
   nir_def *size = nir_image_deref_size(b, size_components, 32, deref,
                                        intr->src[lod_src].ssa,
                                        .image_dim = t->dim,
                                        .image_array = t->array);

   if (t->dim == GLSL_SAMPLER_DIM_CUBE) {
      nir_def *z = t->array ? nir_imul_imm(b, nir_channel(b, size, 2), 6)
                            : nir_imm_int(b, 6);
      size = nir_vec3(b, nir_channel(b, size, 0), nir_channel(b, size, 1), z);
   }

   nir_def *result = nir_ball(b, nir_ult(b, coord, size));
   if (t->dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_image_deref_samples(b, 32, deref,
                                                 .image_dim = t->dim,
                                                 .image_array = t->array);
      result = nir_iand(b, result, nir_ult(b, intr->src[2].ssa, samples));
   }

   return result;
}

static void
record_store(nir_builder *b, const struct tracked_image *t,
             nir_intrinsic_instr *intr, nir_def *index, bool robust_images)
{
   b->cursor = nir_after_instr(&intr->instr);

   nir_def *coord = nir_trim_vector(b, intr->src[1].ssa, coordinate_count(t));
   nir_def *value = nir_format_unpack_rgba(
      b, nir_format_pack_rgba(b, t->format, intr->src[3].ssa), t->format);

   /* A robust store outside the view writes nothing, so it leaves the
    * previous record in place. */
   nir_def *write = robust_images ? in_bounds(b, t, intr, coord)
                                  : nir_imm_true(b);

   nir_store_var(b, t->valid, nir_ior(b, write, nir_load_var(b, t->valid)),
                 0x1);
   nir_store_var(b, t->coord,
                 nir_bcsel(b, write, coord, nir_load_var(b, t->coord)),
                 BITFIELD_MASK(coordinate_count(t)));
   nir_store_var(b, t->value,
                 nir_bcsel(b, write, value, nir_load_var(b, t->value)), 0xf);
   if (t->indexed) {
      nir_store_var(b, t->index,
                    nir_bcsel(b, write, index, nir_load_var(b, t->index)),
                    0x1);
   }
   if (t->dim == GLSL_SAMPLER_DIM_MS) {
      nir_store_var(b, t->sample,
                    nir_bcsel(b, write, intr->src[2].ssa,
                              nir_load_var(b, t->sample)),
                    0x1);
   }
}

static void
forward_load(nir_builder *b, const struct tracked_image *t,
             nir_intrinsic_instr *intr, nir_def *index)
{
   b->cursor = nir_after_instr(&intr->instr);

   nir_def *coord = nir_trim_vector(b, intr->src[1].ssa, coordinate_count(t));
   nir_def *match = nir_iand(
      b, nir_load_var(b, t->valid),
      nir_ball(b, nir_ieq(b, coord, nir_load_var(b, t->coord))));
   if (t->indexed)
      match = nir_iand(b, match, nir_ieq(b, index, nir_load_var(b, t->index)));
   if (t->dim == GLSL_SAMPLER_DIM_MS) {
      match = nir_iand(b, match,
                       nir_ieq(b, intr->src[2].ssa,
                               nir_load_var(b, t->sample)));
   }

   nir_def *result = nir_bcsel(b, match, nir_load_var(b, t->value),
                               &intr->def);
   nir_def_rewrite_uses_after(&intr->def, result);
}

static bool
forward_impl(nir_function_impl *impl, bool robust_images)
{
   struct forward_state state = {0};

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_image_deref_load:
         case nir_intrinsic_image_deref_store:
         case nir_intrinsic_image_deref_atomic:
         case nir_intrinsic_image_deref_atomic_swap:
            gather_access(&state, intr);
            break;
         default:
            break;
         }
      }
   }

   if (state.overflow)
      return false;

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   bool any = false;
   for (unsigned i = 0; i < state.count; ++i) {
      struct tracked_image *t = &state.images[i];
      if (t->unsupported || !t->loaded || !t->stored)
         continue;

      t->valid = nir_local_variable_create(impl, glsl_bool_type(),
                                           "image_store_valid");
      t->coord = nir_local_variable_create(
         impl, glsl_vector_type(GLSL_TYPE_INT, coordinate_count(t)),
         "image_store_coord");
      t->value = nir_local_variable_create(impl, value_type(t),
                                           "image_store_value");
      if (t->indexed) {
         t->index = nir_local_variable_create(impl, glsl_uint_type(),
                                              "image_store_index");
      }
      if (t->dim == GLSL_SAMPLER_DIM_MS) {
         t->sample = nir_local_variable_create(impl, glsl_uint_type(),
                                               "image_store_sample");
      }
      nir_store_var(&b, t->valid, nir_imm_false(&b), 0x1);
      any = true;
   }

   if (!any)
      return false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         /* An image barrier can publish another invocation's writes, or
          * a write through an aliased descriptor. The remembered value is
          * no longer authoritative. Shared-only barriers leave it intact.
          * Do not invalidate at unrelated image writes: forwarding is a
          * correctness workaround for G17P, not just an optimization. */
         if (intr->intrinsic == nir_intrinsic_barrier &&
             (nir_intrinsic_memory_modes(intr) & nir_var_image)) {
            b.cursor = nir_before_instr(instr);
            for (unsigned i = 0; i < state.count; ++i) {
               if (state.images[i].valid)
                  nir_store_var(&b, state.images[i].valid,
                                nir_imm_false(&b), 0x1);
            }
            continue;
         }

         if (intr->intrinsic != nir_intrinsic_image_deref_load &&
             intr->intrinsic != nir_intrinsic_image_deref_store)
            continue;

         nir_def *index;
         nir_variable *image = image_access_variable(intr, &index);
         struct tracked_image *t = NULL;
         for (unsigned i = 0; i < state.count; ++i) {
            if (state.images[i].image == image && state.images[i].valid)
               t = &state.images[i];
         }
         if (!t)
            continue;

         if (intr->intrinsic == nir_intrinsic_image_deref_store)
            record_store(&b, t, intr, index, robust_images);
         else
            forward_load(&b, t, intr, index);
      }
   }

   nir_progress(true, impl, nir_metadata_none);
   return true;
}

bool
hk_nir_forward_image_stores(nir_shader *nir, bool robust_images)
{
   bool progress = false;

   nir_foreach_function_impl(impl, nir)
      progress |= forward_impl(impl, robust_images);

   if (progress)
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   return progress;
}
