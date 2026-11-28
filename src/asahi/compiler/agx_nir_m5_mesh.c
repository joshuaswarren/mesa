/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "agx_nir_m5_mesh.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/u_math.h"
#include <stdlib.h>
#include <string.h>

/* An execution barrier is admitted in a padded workgroup only at the top
 * level of main, where every physical lane can pass it. */
static bool
m5_padded_barrier_ok(const nir_block *block, const nir_intrinsic_instr *i)
{
   return nir_intrinsic_execution_scope(i) == SCOPE_NONE ||
          block->cf_node.parent->type == nir_cf_node_function;
}

/* Run the API body of a padded workgroup under its exact invocation mask
 * (local index < threads). Top-level execution barriers split the body and
 * stay outside the masked segments, so every physical lane passes them, as
 * Apple passes barriers with all lanes enabled; values that cross a barrier
 * are only consumed by the API lanes. */
static void
m5_mask_padded_body(nir_function_impl *entry, unsigned threads)
{
   nir_intrinsic_instr **barriers = NULL;
   unsigned count = 0;
   nir_foreach_block(block, entry) {
      if (block->cf_node.parent->type != nir_cf_node_function)
         continue;
      nir_foreach_instr(in, block) {
         if (in->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
         if (i->intrinsic == nir_intrinsic_barrier &&
             nir_intrinsic_execution_scope(i) != SCOPE_NONE) {
            barriers = realloc(barriers, (count + 1) * sizeof(*barriers));
            barriers[count++] = i;
         }
      }
   }
   nir_cursor start = nir_before_cf_list(&entry->body);
   for (unsigned k = 0; k <= count; ++k) {
      nir_cf_list segment;
      nir_cf_extract(&segment, start,
                     k < count ? nir_before_instr(&barriers[k]->instr)
                               : nir_after_cf_list(&entry->body));
      nir_builder b = nir_builder_at(
         k < count ? nir_before_instr(&barriers[k]->instr)
                   : nir_after_cf_list(&entry->body));
      nir_if *active = nir_push_if(&b, nir_ult_imm(&b,
                          nir_load_local_invocation_index(&b), threads));
      b.cursor = nir_cf_reinsert(&segment, b.cursor);
      nir_pop_if(&b, active);
      if (k < count)
         start = nir_after_instr(&barriers[k]->instr);
   }
   free(barriers);
   if (count)
      nir_repair_ssa_impl(entry);
}

bool
agx_nir_m5_canonicalize_empty_mesh(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_MESH ||
       (s->info.mesh.primitive_type != MESA_PRIM_POINTS &&
        s->info.mesh.primitive_type != MESA_PRIM_LINES) ||
       s->info.outputs_written || s->info.outputs_read)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_intrinsic_instr *count = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_set_vertex_and_primitive_count:
            if (count || nir_intrinsic_stream_id(intr))
               return false;
            for (unsigned i = 0; i < 2; ++i) {
               if (!nir_src_is_const(intr->src[i]) ||
                   nir_src_as_uint(intr->src[i]) != 0)
                  return false;
            }
            count = intr;
            break;
         case nir_intrinsic_store_output:
         case nir_intrinsic_store_per_vertex_output:
         case nir_intrinsic_store_per_primitive_output:
            return false;
         default:
            break;
         }
      }
   }
   if (!count)
      return false;

   nir_metadata_require(impl, nir_metadata_dominance);
   if (!nir_block_dominates(count->instr.block, nir_impl_last_block(impl)))
      return false;

   /* No geometry reaches rasterization. Keep the shader's memory work and
    * use the existing native triangle launch for the proven-empty output. */
   s->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   nir_foreach_function_impl(f, s)
      nir_progress(true, f, nir_metadata_all);
   return true;
}

#define M5_MESH_MAX_PLANES (4 + 128 + 1 + 8)

struct mesh_layout {
   unsigned output_base, primitive_count, vertex_count, position;
   unsigned plane_stride, planes, bytes;
   const struct agx_m5_mesh_varyings *varyings;

   /* Per-primitive outputs (see agx_m5_mesh_unwelds): the API body writes
    * vertex planes to vertex_source and per-primitive planes to
    * primitive_source; the export region at position then holds one
    * private vertex per primitive corner (prim_vertices). */
   bool unweld;
   unsigned export_vertices;
   /* One byte per primitive: nonzero when gl_CullPrimitiveEXT is set. */
   unsigned cull;
   unsigned vertex_source, vertex_stride;
   unsigned primitive_source, primitive_stride;
   int16_t layer_plane, viewport_plane;
   int16_t vertex_plane[M5_MESH_MAX_PLANES], primitive_plane[M5_MESH_MAX_PLANES];

   /* Per-primitive words on the provoking vertex (mesh_layout_provoking):
    * the export region holds the API vertices in place, then private
    * provoking vertices; the body stages primitive j's words in the
    * primitive planes of export slot j. slot_map holds one word per export
    * slot (M5_PV_*). */
   bool provoking;
   unsigned slot_map;

   /* Line meshes with a row bias (m5_mesh_line_bias): one byte per API
    * vertex, nonzero when the vertex ends a line drawn toward +y. */
   unsigned line_marks;
};

/* Export slot map word of the provoking-vertex form: the primitive whose
 * words the slot takes plus one (0: none), the API vertex a private
 * provoking vertex copies, and whether the slot is such a copy. */
#define M5_PV_PRIMITIVE_MASK 0xffffu
#define M5_PV_VERTEX_SHIFT   16
#define M5_PV_COPY           (1u << 24)

/* Index bytes per primitive. Native G17G point and line witnesses
 * (lane-mesh/cwit-meshtopo mpt/mln) store one and two packed index bytes per
 * primitive in the same 16-byte-aligned index region as triangles. */
static unsigned
prim_vertices(const nir_shader *s)
{
   switch (s->info.mesh.primitive_type) {
   case MESA_PRIM_POINTS: return 1;
   case MESA_PRIM_LINES: return 2;
   case MESA_PRIM_TRIANGLES: return 3;
   default: return 0;
   }
}

static bool
agx_m5_mesh_unwelds(const struct agx_m5_mesh_varyings *varyings)
{
   if (varyings->lv)
      return true;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (varyings->primitive_comps[slot])
         return true;
   }
   return false;
}

static struct mesh_layout
mesh_layout(const nir_shader *s, const struct agx_m5_mesh_varyings *varyings)
{
   const unsigned base = ALIGN_POT(s->info.shared_size, 64);
   /* CPU-native maxV3/4/5/8/9/16/32/33/64/256 witnesses establish
    * a 16-byte-aligned SoA plane and aligned triangle-byte/count regions.
    * Native multi-varying witnesses (lane-mesh/native-var objects 9..17)
    * place one plane per UVS word directly after the four position planes,
    * in selector order. */
   const unsigned count =
      ALIGN_POT(prim_vertices(s) * s->info.mesh.max_primitives_out, 16);
   const unsigned position = count + 16;
   const bool unweld = agx_m5_mesh_unwelds(varyings);
   const unsigned export_vertices = unweld
      ? prim_vertices(s) * s->info.mesh.max_primitives_out
      : s->info.mesh.max_vertices_out;
   const unsigned stride = ALIGN_POT(export_vertices, 4) * 4;
   struct mesh_layout layout = {
      .output_base = base, .primitive_count = base + count,
      .vertex_count = base + count + 4, .position = base + position,
      .plane_stride = stride, .planes = varyings->size,
      .bytes = ALIGN_POT(position + varyings->size * stride, 64),
      .varyings = varyings,
      .unweld = unweld,
      .export_vertices = export_vertices,
      .layer_plane = -1, .viewport_plane = -1,
   };
   if (!unweld)
      return layout;

   /* Classify each exported UVS word as a vertex or primitive word. */
   unsigned vertex_planes = 0, primitive_planes = 0;
   for (unsigned plane = 0; plane < M5_MESH_MAX_PLANES; ++plane)
      layout.vertex_plane[plane] = layout.primitive_plane[plane] = -1;
   for (unsigned plane = 0; plane < MIN2(varyings->size, M5_MESH_MAX_PLANES); ++plane) {
      if (varyings->lv && plane == varyings->lv_first)
         continue; /* packed from the layer/viewport words */
      int primitive_slot = -1;
      for (unsigned slot = 0; slot < 64; ++slot) {
         if ((varyings->primitive_comps[slot] ||
              (varyings->lv && slot == VARYING_SLOT_LAYER)) &&
             plane >= varyings->first[slot] &&
             plane < varyings->first[slot] + varyings->components[slot])
            primitive_slot = slot;
      }
      if (primitive_slot < 0)
         layout.vertex_plane[plane] = vertex_planes++;
      else if (varyings->primitive_comps[primitive_slot])
         layout.primitive_plane[plane] = primitive_planes++;
      /* else: the unwritten layer word, exported as zero */
   }
   if (varyings->primitive_comps[VARYING_SLOT_LAYER])
      layout.layer_plane =
         layout.primitive_plane[varyings->first[VARYING_SLOT_LAYER]];
   if (varyings->primitive_comps[VARYING_SLOT_VIEWPORT])
      layout.viewport_plane =
         layout.primitive_plane[varyings->first[VARYING_SLOT_VIEWPORT]];

   layout.vertex_stride = ALIGN_POT(s->info.mesh.max_vertices_out, 4) * 4;
   layout.primitive_stride = ALIGN_POT(s->info.mesh.max_primitives_out, 4) * 4;
   const unsigned vertex_source = position + varyings->size * stride;
   const unsigned primitive_source =
      vertex_source + vertex_planes * layout.vertex_stride;
   layout.vertex_source = base + vertex_source;
   layout.primitive_source = base + primitive_source;
   layout.bytes = ALIGN_POT(primitive_source +
                            primitive_planes * layout.primitive_stride, 64);
   return layout;
}

/* The provoking-vertex form of per-primitive user and primitive ID words,
 * for meshes whose private vertices do not fit (agx_nir_lower_m5_mesh).
 * Export slot k holds API vertex k in place, so per-vertex stores land
 * where the non-per-primitive layout puts them, and the body stages
 * primitive j's words in the primitive planes of slot j; slots past the
 * API vertices take private provoking vertices. The slot count is the
 * largest that fits the threadgroup memory, between max(vertices,
 * primitives) and min(256, vertices + primitives). export_vertices is 0
 * when even the smallest does not fit. Layer and viewport words are not
 * taken: they keep the private-vertex form. */
static struct mesh_layout
mesh_layout_provoking(const nir_shader *s,
                      const struct agx_m5_mesh_varyings *varyings)
{
   const unsigned base = ALIGN_POT(s->info.shared_size, 64);
   const unsigned max_vertices = s->info.mesh.max_vertices_out;
   const unsigned max_primitives = s->info.mesh.max_primitives_out;
   const unsigned count = ALIGN_POT(prim_vertices(s) * max_primitives, 16);
   const unsigned position = count + 16;
   const unsigned cull_bytes =
      varyings->cull_primitive ? ALIGN_POT(max_primitives, 64) : 0;
   struct mesh_layout layout = {
      .output_base = base, .primitive_count = base + count,
      .vertex_count = base + count + 4, .position = base + position,
      .planes = varyings->size,
      .varyings = varyings,
      .layer_plane = -1, .viewport_plane = -1,
      .provoking = true,
   };
   const unsigned smallest = ALIGN_POT(MAX2(max_vertices, max_primitives), 4);
   const unsigned largest = MIN2(256, ALIGN_POT(max_vertices + max_primitives, 4));
   for (unsigned slots = largest; slots >= smallest; slots -= 4) {
      /* Planes, then one map word per slot, then the cull bytes. */
      const unsigned map = position + varyings->size * 4 * slots;
      const unsigned cull = ALIGN_POT(map + 4 * slots, 64);
      if (base + cull + cull_bytes <= 32768) {
         layout.export_vertices = slots;
         layout.plane_stride = 4 * slots;
         layout.slot_map = base + map;
         layout.cull = varyings->cull_primitive ? base + cull : 0;
         layout.bytes = cull + cull_bytes;
         break;
      }
   }
   for (unsigned plane = 0; plane < M5_MESH_MAX_PLANES; ++plane)
      layout.vertex_plane[plane] = layout.primitive_plane[plane] = -1;
   for (unsigned plane = 0; plane < MIN2(varyings->size, M5_MESH_MAX_PLANES); ++plane) {
      bool primitive = false;
      for (unsigned slot = 0; slot < 64; ++slot) {
         primitive |= varyings->primitive_comps[slot] &&
                      plane >= varyings->first[slot] &&
                      plane < varyings->first[slot] + varyings->components[slot];
      }
      /* Primitive words are addressed through primitive_source and
       * primitive_stride like the private-vertex form, here the export
       * planes themselves. */
      if (primitive)
         layout.primitive_plane[plane] = plane;
      else
         layout.vertex_plane[plane] = plane;
   }
   layout.primitive_source = layout.position;
   layout.primitive_stride = layout.plane_stride;
   return layout;
}

/* Shared address of a vertex-output word written by the API body. */
static unsigned
vertex_plane_address(const struct mesh_layout *layout, unsigned plane)
{
   if (!layout->unweld)
      return layout->position + layout->plane_stride * plane;
   assert(layout->vertex_plane[plane] >= 0);
   return layout->vertex_source +
          layout->vertex_stride * layout->vertex_plane[plane];
}

static unsigned
vertex_output_slot(nir_intrinsic_instr *i)
{
   return nir_intrinsic_io_semantics(i).location + nir_src_as_uint(i->src[2]);
}

/* Plane (UVS word) of the first component of a per-vertex output slot. */
static unsigned
first_plane(const struct agx_m5_mesh_varyings *varyings, unsigned slot)
{
   return slot == VARYING_SLOT_POS ? 0 : varyings->first[slot];
}

static nir_def *
widen_output(nir_builder *b, nir_intrinsic_instr *i, nir_def *value)
{
   /* The UVS carries 32-bit words; FP16/INT16 varyings are converted the
    * same way agx_nir_lower_uvs does for the vertex stage, and the fragment
    * linkage (32-bit counts only) narrows them again on load. */
   if (value->bit_size == 32)
      return value;
   nir_alu_type type = nir_alu_type_get_base_type(nir_intrinsic_src_type(i));
   return nir_convert_to_bit_size(b, value, type, 32);
}

static bool
lower(nir_builder *b,nir_intrinsic_instr *i,void *data)
{
   const struct mesh_layout *layout = data;
   const unsigned output_base = layout->output_base;
   switch(i->intrinsic) {
   case nir_intrinsic_store_per_vertex_output: {
      b->cursor=nir_before_instr(&i->instr);
      nir_def *element=nir_imul_imm(b,i->src[1].ssa,4);
      const unsigned slot=vertex_output_slot(i);
      const unsigned first=first_plane(layout->varyings,slot);
      unsigned component=nir_intrinsic_component(i);
      nir_def *value=widen_output(b,i,i->src[0].ssa);
      u_foreach_bit(c,nir_intrinsic_write_mask(i)) {
         nir_def *address=nir_iadd_imm(b,element,
            vertex_plane_address(layout,first+component+c));
         nir_store_shared(b,nir_channel(b,value,c),address,
                            .write_mask=1,.align_mul=4);
         if (slot == VARYING_SLOT_CLIP_DIST0 ||
             slot == VARYING_SLOT_CLIP_DIST1) {
            const unsigned clip =
               4 * (slot - VARYING_SLOT_CLIP_DIST0) + component + c;
            if (clip < layout->varyings->clip_distance_count) {
               nir_def *clip_address = nir_iadd_imm(b, element,
                  vertex_plane_address(layout,
                     layout->varyings->clip_distance_first + clip));
               nir_store_shared(b, nir_channel(b, value, c), clip_address,
                                .write_mask = 1, .align_mul = 4);
            }
         }
      }
      break;
   }
   case nir_intrinsic_store_per_primitive_output: {
      b->cursor=nir_before_instr(&i->instr);
      const unsigned slot = vertex_output_slot(i);
      if (slot == VARYING_SLOT_CULL_PRIMITIVE) {
         nir_def *value = i->src[0].ssa;
         nir_def *culled = value->bit_size == 1 ? value : nir_ine_imm(b, value, 0);
         nir_store_shared(b, nir_b2i8(b, culled),
            nir_iadd_imm(b, i->src[1].ssa, layout->cull), .write_mask = 1,
            .align_mul = 1);
         break;
      }
      if (slot != VARYING_SLOT_PRIMITIVE_INDICES) {
         /* A primitive word: the export below replicates it to the
          * private vertices of its primitive. */
         nir_def *element = nir_imul_imm(b, i->src[1].ssa, 4);
         nir_def *value = widen_output(b, i, i->src[0].ssa);
         u_foreach_bit(c, nir_intrinsic_write_mask(i)) {
            int plane = layout->primitive_plane[layout->varyings->first[slot] +
                                                nir_intrinsic_component(i) + c];
            assert(plane >= 0);
            nir_store_shared(b, nir_channel(b, value, c),
               nir_iadd_imm(b, element, layout->primitive_source +
                                        layout->primitive_stride * plane),
               .write_mask = 1, .align_mul = 4);
         }
         break;
      }
      nir_def *element=nir_imul_imm(b,i->src[1].ssa,
                                    prim_vertices(b->shader));
      unsigned component=nir_intrinsic_component(i);
      u_foreach_bit(c,nir_intrinsic_write_mask(i)) {
         nir_def *index=nir_channel(b,i->src[0].ssa,c);
         /* Native helper creates per-byte validity masks with two ISELs,
          * adds low0xff/high0xff00, then ANDs the original index word.
          * Valid index bytes pass through unchanged. Cull is not mapped. */
         nir_def *packed=nir_u2u8(b,index);
         nir_store_shared(b,packed,nir_iadd_imm(b,element,output_base+component+c),
                            .write_mask=1,.align_mul=1);
      }
      break;
   }
   case nir_intrinsic_set_vertex_and_primitive_count: {
      b->cursor=nir_before_instr(&i->instr);
      nir_push_if(b,nir_ieq_imm(b,nir_load_local_invocation_index(b),0));
      nir_store_shared(b,nir_u2u16(b,i->src[1].ssa),nir_imm_int(b,layout->primitive_count),
                         .write_mask=1,.align_mul=2);
      nir_store_shared(b,nir_u2u16(b,i->src[0].ssa),nir_imm_int(b,layout->vertex_count),
                         .write_mask=1,.align_mul=2);
      nir_pop_if(b,NULL);
      break;
   }
   default:return false;
   }
   nir_instr_remove(&i->instr);
   return true;
}

#define USER_VARYINGS BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32)
#define MESH_VERTEX_VARYINGS \
   (USER_VARYINGS | VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 | \
    BITFIELD64_RANGE(VARYING_SLOT_CULL_PRIMITIVE, 2))
/* Per-primitive outputs exported as user words of private vertices. */
#define MESH_PRIMITIVE_WORDS (USER_VARYINGS | VARYING_BIT_PRIMITIVE_ID)

bool
agx_m5_mesh_gather_varyings(const nir_shader *s,
                            struct agx_m5_mesh_varyings *varyings)
{
   memset(varyings, 0, sizeof(*varyings));
   if (s->info.clip_distance_array_size +
       s->info.cull_distance_array_size > 8)
      return false;
   varyings->clip_distance_count = s->info.clip_distance_array_size;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            if (i->intrinsic != nir_intrinsic_store_per_vertex_output &&
                i->intrinsic != nir_intrinsic_store_per_primitive_output)
               continue;
            if (!nir_src_is_const(i->src[2]))
               return false;
            const unsigned slot = vertex_output_slot(i);
            if (i->intrinsic == nir_intrinsic_store_per_primitive_output) {
               if (slot == VARYING_SLOT_PRIMITIVE_INDICES)
                  continue;
               if (slot == VARYING_SLOT_CULL_PRIMITIVE) {
                  if (nir_intrinsic_component(i) != 0 ||
                      nir_intrinsic_write_mask(i) != 1)
                     return false;
                  varyings->cull_primitive = true;
                  continue;
               }
               /* Per-primitive user words become flat words of the
                * primitive's private vertices; layer/viewport fill the
                * vertex layer/viewport word. The fragment side reads both
                * as flat inputs (agx_nir_lower_m5_mesh_fragment). */
               const unsigned end = nir_intrinsic_component(i) +
                                    util_last_bit(nir_intrinsic_write_mask(i));
               const unsigned bits = i->src[0].ssa->bit_size;
               if (slot >= 64 || end > 4 || (bits != 32 && bits != 16))
                  return false;
               if (USER_VARYINGS & BITFIELD64_BIT(slot)) {
                  varyings->primitive_comps[slot] =
                     MAX2(varyings->primitive_comps[slot], end);
               } else if (slot == VARYING_SLOT_LAYER ||
                          slot == VARYING_SLOT_VIEWPORT) {
                  if (end != 1)
                     return false;
                  varyings->primitive_comps[slot] = 1;
                  varyings->lv = true;
               } else if (slot == VARYING_SLOT_PRIMITIVE_ID) {
                  /* A flat user word of the primitive's private vertices,
                   * which the fragment shader's gl_PrimitiveID links to. */
                  if (end != 1)
                     return false;
                  varyings->primitive_comps[slot] = 1;
               } else {
                  return false;
               }
               continue;
            }
            if (slot == VARYING_SLOT_POS)
               continue;
            /* Point size also reaches the raster state of triangle and line
             * meshes (a shader object's polygon mode can be POINT). */
            if (slot == VARYING_SLOT_PSIZ) {
               if (nir_intrinsic_component(i) != 0 ||
                   nir_intrinsic_write_mask(i) != 1)
                  return false;
               varyings->components[slot] = 1;
               continue;
            }
            if (slot >= 64 || !(MESH_VERTEX_VARYINGS & BITFIELD64_BIT(slot)))
               return false;
            const unsigned end = nir_intrinsic_component(i) +
                                 util_last_bit(nir_intrinsic_write_mask(i));
            if (end > 4)
               return false;
            varyings->components[slot] = MAX2(varyings->components[slot], end);
         }
      }
   }
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (!(MESH_PRIMITIVE_WORDS & BITFIELD64_BIT(slot)) ||
          !varyings->primitive_comps[slot])
         continue;
      /* A location is either per-vertex or per-primitive. */
      if (varyings->components[slot])
         return false;
      varyings->components[slot] = varyings->primitive_comps[slot];
   }
   /* Like the M5 vertex stage (hk_lower_hw_vs with layer_as_varying), the
    * layer is also a flat user word for fragment reads, zero when only the
    * viewport is written, and a written viewport is one as well. */
   if (varyings->lv) {
      varyings->components[VARYING_SLOT_LAYER] = 1;
      varyings->components[VARYING_SLOT_VIEWPORT] =
         varyings->primitive_comps[VARYING_SLOT_VIEWPORT];
   }
   /* Declaration order. A single user slot (or one interpolation class) sits
    * at the base of the user section for every fragment shader; otherwise the
    * fragment shader chooses the class order at draw time. The UVS groups
    * that follow are layer/viewport, then clip distances. */
   unsigned next = 4, slots = 0;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (!varyings->components[slot] || slot == VARYING_SLOT_PSIZ)
         continue;
      varyings->first[slot] = next;
      next += varyings->components[slot];
      slots++;
   }
   /* Point size follows the user words, in the UVS group order of
    * agx_nir_lower_uvs that programs the raster state; with no user words
    * it is the plane after the position, as in native size-only points
    * (cwit-meshtopo mpt). */
   if (varyings->components[VARYING_SLOT_PSIZ])
      varyings->first[VARYING_SLOT_PSIZ] = next++;
   if (varyings->lv)
      varyings->lv_first = next++;
   varyings->clip_distance_first = next;
   varyings->size = next + varyings->clip_distance_count;
   varyings->dynamic = slots > 1;
   return true;
}

/* Every user plane is covered exactly once by the slot placement. */
static bool
varyings_valid(const struct agx_m5_mesh_varyings *v)
{
   const unsigned lv = v->lv ? 1 : 0;
   if (v->size < 4 || v->size > 4 + 128 + 8 + lv ||
       v->clip_distance_count > 8 ||
       v->size < 4 + lv + v->clip_distance_count ||
       (v->clip_distance_count &&
        v->clip_distance_first + v->clip_distance_count != v->size))
      return false;
   /* User words, then the optional layer/viewport word, then clip words. */
   const unsigned user_end = v->size - v->clip_distance_count - lv;
   if (v->lv && v->lv_first != user_end)
      return false;
   uint8_t used[M5_MESH_MAX_PLANES] = {0};
   unsigned user = 0;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (!v->components[slot])
         continue;
      const bool lv_word = v->lv && (slot == VARYING_SLOT_LAYER ||
                                     slot == VARYING_SLOT_VIEWPORT);
      if (lv_word) {
         if (v->components[slot] != 1 ||
             (slot == VARYING_SLOT_VIEWPORT && !v->primitive_comps[slot]))
            return false;
      } else if (!((MESH_VERTEX_VARYINGS | VARYING_BIT_PSIZ |
                    VARYING_BIT_PRIMITIVE_ID) & BITFIELD64_BIT(slot)) ||
                 (slot == VARYING_SLOT_PRIMITIVE_ID &&
                  !v->primitive_comps[slot]) ||
                 (v->primitive_comps[slot] &&
                  (!(MESH_PRIMITIVE_WORDS & BITFIELD64_BIT(slot)) ||
                   v->primitive_comps[slot] != v->components[slot]))) {
         return false;
      }
      if (v->components[slot] > 4 ||
          v->first[slot] < 4 ||
          v->first[slot] + v->components[slot] > user_end)
         return false;
      for (unsigned c = 0; c < v->components[slot]; ++c) {
         if (used[v->first[slot] + c]++)
            return false;
      }
      user += v->components[slot];
   }
   if (v->lv != (v->primitive_comps[VARYING_SLOT_LAYER] ||
                 v->primitive_comps[VARYING_SLOT_VIEWPORT]) ||
       (v->lv && !v->components[VARYING_SLOT_LAYER]))
      return false;
   return user + 4 + lv + v->clip_distance_count == v->size;
}

static void
m5_mesh_rendezvous(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

/* Lines drawn toward +y (agx_m5_mesh_varyings::line_bias_root_B).
 *
 * As on G17P (agx_nir_lower_gen1_mesh), the rasterizer puts a line endpoint
 * that lies exactly on a pixel row boundary into the row above, where
 * Vulkan's diamond is open, so a line drawn toward +y covers rows one too
 * high. Move the vertices of such lines a fraction of a pixel toward +y (the
 * root holds that step in NDC, signed by the viewport's y direction). A
 * vertex shared with a line drawn the other way moves too; off a boundary
 * the step changes nothing.
 *
 * This moves the API vertices where the body stored them, once the body
 * and the cull move are done and before anything copies vertex words, so
 * the private vertices of the private-vertex form and the private
 * provoking copies of the provoking-vertex form take the moved positions.
 * Every lane takes part, with a rendezvous between the steps, since SIMD
 * group 0 can have fewer lanes than there are vertices or primitives; the
 * rendezvous that follows (unweld copy, provoking epilogue or export)
 * publishes the last step.
 */
static void
m5_mesh_line_bias(nir_builder *b, const nir_shader *s,
                  const struct mesh_layout *layout, unsigned root_B)
{
   const unsigned max_vertices = s->info.mesh.max_vertices_out;
   const unsigned max_primitives = s->info.mesh.max_primitives_out;
   const unsigned lanes = s->info.workgroup_size[0] *
                          s->info.workgroup_size[1] *
                          s->info.workgroup_size[2];
   /* Position y and w of API vertex 0. */
   const unsigned y_plane = vertex_plane_address(layout, 1);
   const unsigned w_plane = vertex_plane_address(layout, 3);
   nir_def *index = nir_load_local_invocation_index(b);

   /* Clear the marks. The body never touches them, so this needs no
    * rendezvous before it. */
   const unsigned words = DIV_ROUND_UP(max_vertices, 4);
   for (unsigned w = 0; w < words; w += lanes) {
      nir_def *word = nir_iadd_imm(b, index, w);
      nir_push_if(b, nir_ult_imm(b, word, words));
      nir_store_shared(b, nir_imm_int(b, 0),
         nir_iadd_imm(b, nir_imul_imm(b, word, 4), layout->line_marks),
         .write_mask = 1, .align_mul = 4);
      nir_pop_if(b, NULL);
   }
   m5_mesh_rendezvous(b);

   /* Mark both ends of every kept line drawn toward +y in front of the
    * eye. */
   nir_def *bias = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, nir_load_root_agx(b), root_B),
      .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
   nir_def *primitives = nir_umin(b, nir_u2u32(b, nir_load_shared(b, 1, 16,
      nir_imm_int(b, layout->primitive_count), .align_mul = 2)),
      nir_imm_int(b, max_primitives));
   for (unsigned base = 0; base < max_primitives; base += lanes) {
      nir_def *j = nir_iadd_imm(b, index, base);
      nir_push_if(b, nir_ult(b, j, primitives));
      nir_def *ends[2], *ndc_y[2], *ok = nir_imm_true(b);
      for (unsigned c = 0; c < 2; ++c) {
         /* An out-of-range index stays within the planes and marks. */
         ends[c] = nir_umin(b, nir_u2u32(b, nir_load_shared(b, 1, 8,
            nir_iadd_imm(b, nir_imul_imm(b, j, 2), layout->output_base + c),
            .align_mul = 1)), nir_imm_int(b, max_vertices - 1));
         nir_def *at = nir_imul_imm(b, ends[c], 4);
         nir_def *y = nir_load_shared(b, 1, 32, nir_iadd_imm(b, at, y_plane),
                                      .align_mul = 4);
         nir_def *w = nir_load_shared(b, 1, 32, nir_iadd_imm(b, at, w_plane),
                                      .align_mul = 4);
         ok = nir_iand(b, ok, nir_flt(b, nir_imm_float(b, 0.0f), w));
         ndc_y[c] = nir_fdiv(b, y, w);
      }
      nir_def *toward = nir_flt(b, nir_imm_float(b, 0.0f),
         nir_fmul(b, nir_fsub(b, ndc_y[1], ndc_y[0]), bias));
      nir_push_if(b, nir_iand(b, ok, toward));
      for (unsigned c = 0; c < 2; ++c) {
         nir_store_shared(b, nir_imm_intN_t(b, 1, 8),
                          nir_iadd_imm(b, ends[c], layout->line_marks),
                          .write_mask = 1, .align_mul = 1);
      }
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
   }
   m5_mesh_rendezvous(b);

   /* Move every marked API vertex: y += step * w. */
   nir_def *vertices = nir_umin(b, nir_u2u32(b, nir_load_shared(b, 1, 16,
      nir_imm_int(b, layout->vertex_count), .align_mul = 2)),
      nir_imm_int(b, max_vertices));
   for (unsigned base = 0; base < max_vertices; base += lanes) {
      nir_def *i = nir_iadd_imm(b, index, base);
      nir_push_if(b, nir_ult(b, i, vertices));
      nir_def *marked = nir_u2u32(b, nir_load_shared(b, 1, 8,
         nir_iadd_imm(b, i, layout->line_marks), .align_mul = 1));
      nir_push_if(b, nir_ine_imm(b, marked, 0));
      nir_def *at = nir_imul_imm(b, i, 4);
      nir_def *y = nir_load_shared(b, 1, 32, nir_iadd_imm(b, at, y_plane),
                                   .align_mul = 4);
      nir_def *w = nir_load_shared(b, 1, 32, nir_iadd_imm(b, at, w_plane),
                                   .align_mul = 4);
      nir_store_shared(b, nir_ffma(b, bias, w, y), nir_iadd_imm(b, at, y_plane),
                       .write_mask = 1, .align_mul = 4);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
   }
}

/* Per-primitive words on the provoking vertex (mesh_layout_provoking).
 *
 * A native M5 mesh draw takes flat inputs from the vertex its PPP
 * flat-shading selector names, the draw's provoking vertex (see the native
 * mesh note at fields.flat_shading_vertex in hk_cmd_draw.c), and the draw
 * root carries the same selector (load_provoking_last). Per-primitive words
 * are flat inputs (agx_nir_lower_m5_mesh_fragment), so only the provoking
 * vertex of a primitive needs them; its other corners keep naming API
 * vertices, whose vertex words every copy shares.
 *
 * In primitive order, invocation 0 gives every primitive a provoking export
 * slot: the API vertex itself when no earlier primitive took it or took it
 * with the same words; otherwise the next slot past the API vertices, a
 * private copy of that vertex, which the primitive's provoking index byte
 * then names. SIMD group 0 then copies the vertex words of the private
 * copies and moves into every slot the primitive words staged at the slot
 * of its primitive, one plane at a time: all lanes read before a rendezvous
 * and write after it, so no move reads a word another move replaced.
 * Primitive order, winding and interpolated values are unchanged. Once the
 * slots run out, a primitive keeps the shared provoking vertex and with it
 * the words of the primitive that took the vertex first.
 */
static void
m5_mesh_provoking_epilogue(nir_builder *b, const nir_shader *s,
                           nir_function_impl *impl,
                           const struct mesh_layout *layout)
{
   const unsigned corners = prim_vertices(s);
   const unsigned max_vertices = s->info.mesh.max_vertices_out;
   const unsigned max_primitives = s->info.mesh.max_primitives_out;
   const unsigned slots = layout->export_vertices;
   const unsigned lanes = s->info.workgroup_size[0] *
                          s->info.workgroup_size[1] *
                          s->info.workgroup_size[2];
   unsigned primitive_planes[M5_MESH_MAX_PLANES], vertex_planes[M5_MESH_MAX_PLANES];
   unsigned np = 0, nv = 0;
   for (unsigned plane = 0; plane < MIN2(layout->planes, M5_MESH_MAX_PLANES); ++plane) {
      if (layout->primitive_plane[plane] >= 0)
         primitive_planes[np++] = plane;
      else
         vertex_planes[nv++] = plane;
   }

   /* Every API output store (and cull move) is done: clear the slot map. */
   m5_mesh_rendezvous(b);
   nir_def *index = nir_load_local_invocation_index(b);
   for (unsigned w = 0; w < slots; w += lanes) {
      nir_def *slot = nir_iadd_imm(b, index, w);
      nir_push_if(b, nir_ult_imm(b, slot, slots));
      nir_store_shared(b, nir_imm_int(b, 0),
         nir_iadd_imm(b, nir_imul_imm(b, slot, 4), layout->slot_map),
         .write_mask = 1, .align_mul = 4);
      nir_pop_if(b, NULL);
   }
   m5_mesh_rendezvous(b);

   nir_variable *jv = nir_local_variable_create(impl, glsl_uint_type(), "pv_j");
   nir_variable *kv = nir_local_variable_create(impl, glsl_uint_type(), "pv_k");
   nir_push_if(b, nir_ieq_imm(b, index, 0));
   {
      nir_def *primitives = nir_umin(b, nir_u2u32(b, nir_load_shared(b, 1, 16,
         nir_imm_int(b, layout->primitive_count), .align_mul = 2)),
         nir_imm_int(b, max_primitives));
      nir_def *vertices = nir_umin(b, nir_u2u32(b, nir_load_shared(b, 1, 16,
         nir_imm_int(b, layout->vertex_count), .align_mul = 2)),
         nir_imm_int(b, max_vertices));
      /* An out-of-range index names the last API vertex, never a copy. */
      nir_def *last = nir_iadd_imm(b, nir_umax(b, vertices, nir_imm_int(b, 1)), -1);
      nir_def *corner = nir_imm_int(b, 0);
      if (corners > 1) {
         corner = nir_bcsel(b, nir_ine_imm(b, nir_load_provoking_last(b), 0),
                            nir_imm_int(b, corners - 1), corner);
      }
      nir_store_var(b, jv, nir_imm_int(b, 0), 1);
      nir_store_var(b, kv, vertices, 1);
      nir_push_loop(b);
      {
         nir_def *j = nir_load_var(b, jv);
         nir_break_if(b, nir_uge(b, j, primitives));
         nir_def *at = nir_iadd(b, nir_imul_imm(b, j, corners),
                                nir_iadd_imm(b, corner, layout->output_base));
         nir_def *vertex = nir_umin(b, nir_u2u32(b, nir_load_shared(b, 1, 8, at,
            .align_mul = 1)), last);
         nir_def *entry = nir_iadd_imm(b, nir_imul_imm(b, vertex, 4),
                                       layout->slot_map);
         nir_def *owner = nir_iand_imm(b, nir_load_shared(b, 1, 32, entry,
            .align_mul = 4), M5_PV_PRIMITIVE_MASK);
         nir_def *tag = nir_iadd_imm(b, j, 1);
         nir_push_if(b, nir_ieq_imm(b, owner, 0));
         {
            nir_store_shared(b, tag, entry, .write_mask = 1, .align_mul = 4);
         }
         nir_push_else(b, NULL);
         {
            nir_def *mine = nir_imul_imm(b, j, 4);
            nir_def *theirs = nir_imul_imm(b, nir_iadd_imm(b, owner, -1), 4);
            nir_def *same = nir_imm_true(b);
            for (unsigned q = 0; q < np; ++q) {
               const unsigned base = layout->primitive_source +
                                     layout->primitive_stride * primitive_planes[q];
               same = nir_iand(b, same, nir_ieq(b,
                  nir_load_shared(b, 1, 32, nir_iadd_imm(b, mine, base), .align_mul = 4),
                  nir_load_shared(b, 1, 32, nir_iadd_imm(b, theirs, base), .align_mul = 4)));
            }
            nir_def *k = nir_load_var(b, kv);
            nir_push_if(b, nir_iand(b, nir_inot(b, same), nir_ult_imm(b, k, slots)));
            {
               nir_store_shared(b, nir_ior_imm(b, nir_ior(b, tag,
                  nir_ishl_imm(b, vertex, M5_PV_VERTEX_SHIFT)), M5_PV_COPY),
                  nir_iadd_imm(b, nir_imul_imm(b, k, 4), layout->slot_map),
                  .write_mask = 1, .align_mul = 4);
               nir_store_shared(b, nir_u2u8(b, k), at, .write_mask = 1,
                                .align_mul = 1);
               nir_store_var(b, kv, nir_iadd_imm(b, k, 1), 1);
            }
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
         nir_store_var(b, jv, nir_iadd_imm(b, j, 1), 1);
      }
      nir_pop_loop(b, NULL);
      /* The exported vertex count: API vertices plus private copies. */
      nir_store_shared(b, nir_u2u16(b, nir_load_var(b, kv)),
         nir_imm_int(b, layout->vertex_count), .write_mask = 1, .align_mul = 2);
   }
   nir_pop_if(b, NULL);
   m5_mesh_rendezvous(b);

   /* SIMD group 0, slot lane + 32t: move[t] is the byte offset of the
    * staging slot whose primitive words the slot takes, or ~0. Step 0
    * copies vertex words and reads the first primitive plane; step n
    * writes plane n - 1 and reads plane n, which no write of the step
    * touches. */
   const unsigned batches = DIV_ROUND_UP(slots, 32);
   nir_variable *move[256 / 32], *word[256 / 32];
   assert(batches <= ARRAY_SIZE(move));
   for (unsigned t = 0; t < batches; ++t) {
      move[t] = nir_local_variable_create(impl, glsl_uint_type(), "pv_move");
      word[t] = nir_local_variable_create(impl, glsl_uint_type(), "pv_word");
   }
   for (unsigned step = 0; step <= np; ++step) {
      nir_push_if(b, nir_ieq_imm(b, nir_load_subgroup_id(b), 0));
      nir_def *lane = nir_load_subgroup_invocation(b);
      nir_def *exported = step == 0 ? nir_u2u32(b, nir_load_shared(b, 1, 16,
         nir_imm_int(b, layout->vertex_count), .align_mul = 2)) : NULL;
      for (unsigned t = 0; t < batches; ++t) {
         nir_def *slot = nir_iadd_imm(b, lane, 32 * t);
         nir_def *target = nir_imul_imm(b, slot, 4);
         if (step == 0) {
            nir_store_var(b, move[t], nir_imm_int(b, -1), 1);
            nir_push_if(b, nir_ult(b, slot, exported));
            {
               nir_def *entry = nir_load_shared(b, 1, 32,
                  nir_iadd_imm(b, target, layout->slot_map), .align_mul = 4);
               nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, entry, M5_PV_COPY), 0));
               {
                  /* Private copies sit past the API vertices they copy. */
                  nir_def *source = nir_imul_imm(b, nir_iand_imm(b,
                     nir_ushr_imm(b, entry, M5_PV_VERTEX_SHIFT), 0xff), 4);
                  for (unsigned q = 0; q < nv; ++q) {
                     const unsigned base = layout->position +
                                           layout->plane_stride * vertex_planes[q];
                     nir_store_shared(b, nir_load_shared(b, 1, 32,
                        nir_iadd_imm(b, source, base), .align_mul = 4),
                        nir_iadd_imm(b, target, base), .write_mask = 1,
                        .align_mul = 4);
                  }
               }
               nir_pop_if(b, NULL);
               nir_def *tag = nir_iand_imm(b, entry, M5_PV_PRIMITIVE_MASK);
               nir_def *from = nir_iadd_imm(b, tag, -1);
               nir_push_if(b, nir_iand(b, nir_ine_imm(b, tag, 0),
                                       nir_ine(b, from, slot)));
               nir_store_var(b, move[t], nir_imul_imm(b, from, 4), 1);
               nir_pop_if(b, NULL);
            }
            nir_pop_if(b, NULL);
         } else {
            const unsigned base = layout->primitive_source +
               layout->primitive_stride * primitive_planes[step - 1];
            nir_push_if(b, nir_ine_imm(b, nir_load_var(b, move[t]), UINT32_MAX));
            nir_store_shared(b, nir_load_var(b, word[t]),
               nir_iadd_imm(b, target, base), .write_mask = 1, .align_mul = 4);
            nir_pop_if(b, NULL);
         }
         if (step < np) {
            const unsigned base = layout->primitive_source +
               layout->primitive_stride * primitive_planes[step];
            nir_def *from = nir_load_var(b, move[t]);
            nir_push_if(b, nir_ine_imm(b, from, UINT32_MAX));
            nir_store_var(b, word[t], nir_load_shared(b, 1, 32,
               nir_iadd_imm(b, from, base), .align_mul = 4), 1);
            nir_pop_if(b, NULL);
         }
      }
      nir_pop_if(b, NULL);
      if (step < np)
         m5_mesh_rendezvous(b);
   }
}

bool
agx_nir_lower_m5_mesh_triangle_probe(nir_shader *s)
{
   struct agx_m5_mesh_varyings varyings;
   if (!agx_m5_mesh_gather_varyings(s, &varyings))
      return false;
   return agx_nir_lower_m5_mesh(s, &varyings);
}

bool
agx_nir_lower_m5_mesh(nir_shader *s, const struct agx_m5_mesh_varyings *varyings)
{
   const uint64_t primitive_outputs =
      s->info.per_primitive_outputs & s->info.outputs_written;
   uint64_t primitive_slots = VARYING_BIT_PRIMITIVE_INDICES;
   for (unsigned slot = 0; slot < 64; ++slot) {
      if (varyings->primitive_comps[slot])
         primitive_slots |= BITFIELD64_BIT(slot);
   }
   if (varyings->cull_primitive)
      primitive_slots |= VARYING_BIT_CULL_PRIMITIVE;
   const bool unweld = agx_m5_mesh_unwelds(varyings);
   /* Without layer or viewport words, per-primitive words can instead
    * travel on the provoking vertex alone (mesh_layout_provoking), where
    * private vertices for every corner do not fit. */
   const bool provoking_ok = unweld && !varyings->lv;
   unsigned threads=s->info.workgroup_size[0]*s->info.workgroup_size[1]*
                    s->info.workgroup_size[2];
   if(s->info.stage!=MESA_SHADER_MESH || s->info.shared_size>32768 ||
      s->info.workgroup_size_variable || threads==0 || threads>1024 ||
      s->info.mesh.max_vertices_out==0 || s->info.mesh.max_vertices_out>256 ||
      s->info.mesh.max_primitives_out==0 || s->info.mesh.max_primitives_out>256 ||
      prim_vertices(s)==0 ||
      /* Points carry their size; a point size word follows the user words. */
      (s->info.mesh.primitive_type==MESA_PRIM_POINTS &&
       !varyings->components[VARYING_SLOT_PSIZ]) ||
      (varyings->components[VARYING_SLOT_PSIZ] &&
       varyings->first[VARYING_SLOT_PSIZ] + 1 !=
          varyings->size - varyings->clip_distance_count -
          (varyings->lv ? 1 : 0)) ||
      /* Interstage compaction can leave per_primitive_outputs stale; with
       * per-primitive words every store is checked against them below. */
      (primitive_outputs&~(unweld ? ~UINT64_C(0) : primitive_slots)) ||
      s->info.outputs_written & ~(VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES|MESH_VERTEX_VARYINGS|
                                  (primitive_slots & (VARYING_BIT_LAYER|VARYING_BIT_VIEWPORT|
                                                      VARYING_BIT_CULL_PRIMITIVE|
                                                      VARYING_BIT_PRIMITIVE_ID)) |
                                  VARYING_BIT_PSIZ) ||
      /* Every primitive gets private vertices within the native
       * 256-vertex output limit, or its provoking vertex carries its words. */
      (unweld && !provoking_ok &&
       prim_vertices(s) * s->info.mesh.max_primitives_out > 256) ||
      varyings->clip_distance_count != s->info.clip_distance_array_size ||
      !varyings_valid(varyings))
      return false;
   bool provoking = provoking_ok &&
      prim_vertices(s) * s->info.mesh.max_primitives_out > 256;
   struct mesh_layout layout = provoking ? mesh_layout_provoking(s, varyings)
                                         : mesh_layout(s, varyings);
   if (!provoking && varyings->cull_primitive) {
      layout.cull = layout.output_base + layout.bytes;
      layout.bytes += ALIGN_POT(s->info.mesh.max_primitives_out, 64);
   }
   /* Private vertices hold every per-primitive word twice (source and
    * export copy); the provoking-vertex form holds it once. */
   if (!provoking && provoking_ok &&
       layout.output_base + layout.bytes > 32768) {
      provoking = true;
      layout = mesh_layout_provoking(s, varyings);
   }
   if (provoking ? !layout.export_vertices
                 : layout.output_base + layout.bytes > 32768)
      return false;
   /* The row bias of lines (m5_mesh_line_bias) leaves the output form and
    * slot count as they are. The provoking-vertex form keeps its marks in
    * the slot map, which nothing uses before the epilogue clears it; the
    * other forms append them where they fit, and export the lines unmoved
    * where they do not. */
   bool line_bias = varyings->line_bias_root_B && prim_vertices(s) == 2;
   if (line_bias && layout.provoking) {
      assert(layout.export_vertices >= s->info.mesh.max_vertices_out);
      layout.line_marks = layout.slot_map;
   } else if (line_bias) {
      const unsigned marks = ALIGN_POT(s->info.mesh.max_vertices_out, 64);
      line_bias = layout.output_base + layout.bytes + marks <= 32768;
      if (line_bias) {
         layout.line_marks = layout.output_base + layout.bytes;
         layout.bytes += marks;
      }
   }
   unsigned physical_threads=MAX2(threads,MIN2(layout.export_vertices,32));
   bool counts=false;
   nir_foreach_function_impl(impl,s) {
      nir_foreach_block(block,impl) {
         nir_foreach_instr(in,block) {
            if(in->type!=nir_instr_type_intrinsic)continue;
            nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
            switch(i->intrinsic) {
            case nir_intrinsic_set_vertex_and_primitive_count:counts=true;break;
            case nir_intrinsic_barrier:
               /* Padding moves the API body under its exact active mask;
                * see m5_mask_padded_body. */
               if (physical_threads != threads &&
                   !m5_padded_barrier_ok(block, i))
                  return false;
               break;
            case nir_intrinsic_store_per_vertex_output: {
               if (!nir_src_is_const(i->src[2]))
                  return false;
               const unsigned slot=vertex_output_slot(i);
               const unsigned bits=i->src[0].ssa->bit_size;
               const unsigned end=nir_intrinsic_component(i)+
                                  util_last_bit(nir_intrinsic_write_mask(i));
               if(slot>=64 || (bits!=32 && bits!=16) || end>4)
                  return false;
               if(slot==VARYING_SLOT_POS) {
                  if(bits!=32)return false;
               } else if(!varyings->components[slot] ||
                         end>varyings->components[slot]) {
                  return false;
               }
               break;
            }
            case nir_intrinsic_store_per_primitive_output: {
               if (!nir_src_is_const(i->src[2]))
                  return false;
               const unsigned slot=vertex_output_slot(i);
               const unsigned bits=i->src[0].ssa->bit_size;
               const unsigned end=nir_intrinsic_component(i)+
                                  util_last_bit(nir_intrinsic_write_mask(i));
               if (slot==VARYING_SLOT_CULL_PRIMITIVE) {
                  if (!varyings->cull_primitive || !nir_src_is_const(i->src[2]))
                     return false;
               } else if (slot==VARYING_SLOT_PRIMITIVE_INDICES) {
                  if(nir_intrinsic_io_semantics(i).location!=VARYING_SLOT_PRIMITIVE_INDICES ||
                     nir_intrinsic_component(i)+i->src[0].ssa->num_components>prim_vertices(s) ||
                     nir_src_as_uint(i->src[2])!=0)
                     return false;
               } else if (slot>=64 || !varyings->primitive_comps[slot] ||
                          end>varyings->primitive_comps[slot] ||
                          (bits!=32 && bits!=16)) {
                  return false;
               }
               break;
            }
            case nir_intrinsic_load_per_vertex_output:
            case nir_intrinsic_load_per_primitive_output:
            case nir_intrinsic_launch_mesh_workgroups:return false;
            default:break;
            }
         }
      }
   }
   if (!counts) {
      if (s->info.outputs_written || s->info.outputs_read)
         return false;
      nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   }
   if (physical_threads != threads || s->info.workgroup_size[2] > 1023) {
      /* Native output batches target vertex_base + physical lane. Pad at
       * most the first output SIMD group; later batches reuse it with the
       * captured high16 vertex_base field. Keep the original API body mask. */
      nir_function_impl *entry=nir_shader_get_entrypoint(s);
      nir_foreach_block(block,entry) {
         nir_foreach_instr_safe(in,block) {
            if (in->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
            nir_builder b=nir_builder_at(nir_before_instr(in));
            nir_def *replacement=NULL;
            if (i->intrinsic == nir_intrinsic_load_workgroup_size) {
               replacement=nir_imm_ivec3(&b,s->info.workgroup_size[0],
                  s->info.workgroup_size[1],s->info.workgroup_size[2]);
            } else if (i->intrinsic == nir_intrinsic_load_local_invocation_id ||
                       i->intrinsic == nir_intrinsic_load_global_invocation_id) {
               nir_def *index=nir_load_local_invocation_index(&b);
               unsigned x=s->info.workgroup_size[0], y=s->info.workgroup_size[1];
               replacement=nir_vec3(&b,nir_umod_imm(&b,index,x),
                  nir_umod_imm(&b,nir_udiv_imm(&b,index,x),y),
                  nir_udiv_imm(&b,index,x*y));
               if (i->intrinsic == nir_intrinsic_load_global_invocation_id) {
                  /* The hardware thread position in grid counts the padded
                   * physical workgroup; the API id uses the declared size. */
                  nir_def *size=nir_imm_ivec3(&b,s->info.workgroup_size[0],
                     s->info.workgroup_size[1],s->info.workgroup_size[2]);
                  replacement=nir_iadd(&b,nir_imul(&b,nir_load_workgroup_id(&b),size),
                                       replacement);
                  replacement=nir_u2uN(&b,replacement,i->def.bit_size);
               }
            }
            if (replacement) {
               nir_def_rewrite_uses(&i->def,replacement);
               nir_instr_remove(in);
            }
         }
      }
      /* The payload-entry barrier stays outside the API body mask: native
       * mesh main functions pass it with every lane before the thread-index
       * branch (lane-mesh nwit mpay, nwit2 mtpg). */
      bool entry_barrier = false;
      nir_foreach_instr_safe(in, nir_start_block(entry)) {
         if (in->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(in)->intrinsic ==
                nir_intrinsic_mesh_payload_entry_barrier_agx) {
            nir_instr_remove(in);
            entry_barrier = true;
         }
      }
      m5_mask_padded_body(entry, threads);
      if (entry_barrier) {
         nir_builder b=nir_builder_at(nir_before_impl(entry));
         nir_mesh_payload_entry_barrier_agx(&b);
      }
      s->info.workgroup_size[0]=physical_threads;
      s->info.workgroup_size[1]=s->info.workgroup_size[2]=1;
   }
   /* Keep application shared storage disjoint from the native output arena.
    * The UVST descriptor carries a byte address, so relocation changes no
    * primitive/position plane layout. Existing barriers preserve visibility. */
   nir_shader_intrinsics_pass(s,lower,nir_metadata_none,&layout);
   nir_function_impl *impl=nir_shader_get_entrypoint(s);
   const unsigned max_primitives = s->info.mesh.max_primitives_out;
   if (varyings->cull_primitive) {
      /* An unwritten gl_CullPrimitiveEXT keeps its primitive: clear the cull
       * bytes before the body writes any. */
      nir_builder z = nir_builder_at(nir_before_impl(impl));
      const unsigned words = ALIGN_POT(max_primitives, 64) / 4;
      const unsigned lanes = s->info.workgroup_size[0] *
                             s->info.workgroup_size[1] *
                             s->info.workgroup_size[2];
      nir_def *lane = nir_load_local_invocation_index(&z);
      for (unsigned w = 0; w < words; w += lanes) {
         nir_def *word = nir_iadd_imm(&z, lane, w);
         nir_push_if(&z, nir_ult_imm(&z, word, words));
         nir_store_shared(&z, nir_imm_int(&z, 0),
            nir_iadd_imm(&z, nir_imul_imm(&z, word, 4), layout.cull),
            .write_mask = 1, .align_mul = 4);
         nir_pop_if(&z, NULL);
      }
      nir_barrier(&z, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);
   }
   nir_builder b=nir_builder_at(nir_after_cf_list(&impl->body));
   if (varyings->cull_primitive) {
      /* Drop culled primitives before the export: one lane moves the index
       * bytes (and the words of unwelded primitives) of every kept
       * primitive down, in API order, and stores the new primitive count.
       * The rendezvous that follows (unweld copy or export) publishes them. */
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);
      const unsigned corners = prim_vertices(s);
      unsigned primitive_planes = 0;
      for (unsigned plane = 0; plane < M5_MESH_MAX_PLANES; ++plane)
         primitive_planes = MAX2(primitive_planes, layout.primitive_plane[plane] + 1);
      nir_variable *jv = nir_local_variable_create(impl, glsl_uint_type(), "cull_j");
      nir_variable *kv = nir_local_variable_create(impl, glsl_uint_type(), "cull_k");
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
      nir_store_var(&b, jv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
      nir_def *count = nir_umin(&b, nir_u2u32(&b, nir_load_shared(&b, 1, 16,
         nir_imm_int(&b, layout.primitive_count), .align_mul = 2)),
         nir_imm_int(&b, max_primitives));
      nir_push_loop(&b);
      {
         nir_def *j = nir_load_var(&b, jv);
         nir_break_if(&b, nir_uge(&b, j, count));
         nir_def *culled = nir_load_shared(&b, 1, 8,
            nir_iadd_imm(&b, j, layout.cull), .align_mul = 1);
         nir_push_if(&b, nir_ieq_imm(&b, culled, 0));
         {
            nir_def *k = nir_load_var(&b, kv);
            for (unsigned c = 0; c < corners; ++c) {
               nir_def *index = nir_load_shared(&b, 1, 8, nir_iadd_imm(&b,
                  nir_imul_imm(&b, j, corners), layout.output_base + c), .align_mul = 1);
               nir_store_shared(&b, index, nir_iadd_imm(&b,
                  nir_imul_imm(&b, k, corners), layout.output_base + c),
                  .write_mask = 1, .align_mul = 1);
            }
            for (unsigned plane = 0; layout.unweld && plane < primitive_planes; ++plane) {
               const unsigned base = layout.primitive_source + layout.primitive_stride * plane;
               nir_def *word = nir_load_shared(&b, 1, 32,
                  nir_iadd_imm(&b, nir_imul_imm(&b, j, 4), base), .align_mul = 4);
               nir_store_shared(&b, word, nir_iadd_imm(&b, nir_imul_imm(&b, k, 4), base),
                  .write_mask = 1, .align_mul = 4);
            }
            /* The provoking-vertex form stages primitive words in the
             * primitive planes of the export region (vertex planes stay). */
            for (unsigned plane = 0; layout.provoking && plane < layout.planes; ++plane) {
               if (layout.primitive_plane[plane] < 0)
                  continue;
               const unsigned base = layout.primitive_source + layout.primitive_stride * plane;
               nir_def *word = nir_load_shared(&b, 1, 32,
                  nir_iadd_imm(&b, nir_imul_imm(&b, j, 4), base), .align_mul = 4);
               nir_store_shared(&b, word, nir_iadd_imm(&b, nir_imul_imm(&b, k, 4), base),
                  .write_mask = 1, .align_mul = 4);
            }
            nir_store_var(&b, kv, nir_iadd_imm(&b, k, 1), 1);
         }
         nir_pop_if(&b, NULL);
         nir_store_var(&b, jv, nir_iadd_imm(&b, j, 1), 1);
      }
      nir_pop_loop(&b, NULL);
      nir_store_shared(&b, nir_u2u16(&b, nir_load_var(&b, kv)),
         nir_imm_int(&b, layout.primitive_count), .write_mask = 1, .align_mul = 2);
      nir_pop_if(&b, NULL);
   }
   if (line_bias)
      m5_mesh_line_bias(&b, s, &layout, varyings->line_bias_root_B);
   if (layout.unweld) {
      /* Per-primitive words have no native vertex form here. Give every
       * primitive private vertices: copy each referenced vertex's
       * words, replicate the primitive's words (flat on every corner, so
       * any provoking vertex selects them), and rewrite the index list to
       * the identity. Primitive order, winding and the interpolated values
       * are unchanged. SIMD group 0 copies, in the same 32-lane batches
       * as the export, between two rendezvous: UVB stores read only words
       * published by a threadgroup barrier, as for API output stores (a
       * SIMD-group barrier between the copy and the export left stale
       * words on hardware). Each lane reads and then rewrites only the
       * index bytes of its own private vertices. */
      nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
                     .memory_semantics=NIR_MEMORY_ACQ_REL,.memory_modes=nir_var_mem_shared);
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_subgroup_id(&b), 0));
      nir_def *primitives = nir_umin(&b, nir_u2u32(&b, nir_load_shared(&b, 1, 16,
         nir_imm_int(&b, layout.primitive_count), .align_mul = 2)),
         nir_imm_int(&b, max_primitives));
      const unsigned corners = prim_vertices(s);
      nir_def *vertices = nir_imul_imm(&b, primitives, corners);
      nir_def *lane = nir_load_subgroup_invocation(&b);
      for (unsigned base = 0; base < corners * max_primitives; base += 32) {
         nir_def *j = nir_iadd_imm(&b, lane, base);
         nir_push_if(&b, nir_ult(&b, j, vertices));
         nir_def *index = nir_u2u32(&b, nir_load_shared(&b, 1, 8,
            nir_iadd_imm(&b, j, layout.output_base), .align_mul = 1));
         nir_def *source = nir_imul_imm(&b,
            nir_umin(&b, index, nir_imm_int(&b, s->info.mesh.max_vertices_out - 1)), 4);
         nir_def *primitive = nir_imul_imm(&b, nir_udiv_imm(&b, j, corners), 4);
         nir_def *target = nir_imul_imm(&b, j, 4);
         for (unsigned plane = 0; plane < layout.planes; ++plane) {
            nir_def *value;
            if (varyings->lv && plane == varyings->lv_first) {
               nir_def *layer = nir_imm_int(&b, 0), *viewport = nir_imm_int(&b, 0);
               if (layout.layer_plane >= 0)
                  layer = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, primitive,
                     layout.primitive_source + layout.primitive_stride * layout.layer_plane),
                     .align_mul = 4);
               if (layout.viewport_plane >= 0)
                  viewport = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, primitive,
                     layout.primitive_source + layout.primitive_stride * layout.viewport_plane),
                     .align_mul = 4);
               /* render_target_array_index low, viewport_array_index high,
                * as agx_nir_lower_uvs packs the vertex-stage word. */
               value = nir_pack_32_2x16_split(&b, nir_u2u16(&b, layer),
                                              nir_u2u16(&b, viewport));
            } else if (layout.vertex_plane[plane] >= 0) {
               value = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, source,
                  layout.vertex_source + layout.vertex_stride * layout.vertex_plane[plane]),
                  .align_mul = 4);
            } else if (layout.primitive_plane[plane] < 0) {
               value = nir_imm_int(&b, 0);
            } else {
               value = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, primitive,
                  layout.primitive_source +
                  layout.primitive_stride * layout.primitive_plane[plane]),
                  .align_mul = 4);
            }
            nir_store_shared(&b, value, nir_iadd_imm(&b, target,
               layout.position + layout.plane_stride * plane), .write_mask = 1,
               .align_mul = 4);
         }
         nir_store_shared(&b, nir_u2u8(&b, j),
            nir_iadd_imm(&b, j, layout.output_base), .write_mask = 1,
            .align_mul = 1);
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);
   }
   if (layout.provoking)
      m5_mesh_provoking_epilogue(&b, s, impl, &layout);
   nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
                  .memory_semantics=NIR_MEMORY_ACQ_REL,.memory_modes=nir_var_mem_shared);
   /* Native write_uvb helpers execute on SIMD group0 after rendezvous. */
   nir_push_if(&b,nir_ieq_imm(&b,nir_load_subgroup_id(&b),0));
   nir_def *p=nir_load_shared(&b,1,16,nir_imm_int(&b,layout.primitive_count),.align_mul=2);
   nir_def *v=nir_load_shared(&b,1,16,nir_imm_int(&b,layout.vertex_count),.align_mul=2);
   if (layout.unweld) {
      p = nir_umin(&b, p, nir_imm_intN_t(&b, max_primitives, 16));
      v = nir_imul_imm(&b, p, prim_vertices(s));
   } else if (layout.provoking) {
      /* The epilogue stored the exported vertex count, at most the slots. */
      p = nir_umin(&b, p, nir_imm_intN_t(&b, max_primitives, 16));
   }
   nir_allocate_mesh_uvb_agx(&b,p,v);
   nir_def *lane=nir_load_subgroup_invocation(&b);
   if (varyings->clip_m1_1) {
      /* As agx_nir_lower_clip_m1_1, on the exported z and w. */
      const unsigned z_plane = first_plane(varyings, VARYING_SLOT_POS) + 2;
      const uint32_t fp = b.fp_math_ctrl;
      b.fp_math_ctrl = nir_fp_no_reassoc | nir_fp_no_transform;
      nir_def *c = nir_load_clip_z_coeff(&b);
      for (unsigned vertex = 0; vertex < layout.export_vertices; vertex += 32) {
         nir_def *index = nir_iadd_imm(&b, lane, vertex);
         nir_push_if(&b, nir_ult(&b, index, nir_u2u32(&b, v)));
         nir_def *addr = nir_iadd_imm(&b, nir_imul_imm(&b, index, 4),
                                      layout.position + layout.plane_stride * z_plane);
         nir_def *z = nir_load_shared(&b, 1, 32, addr, .align_mul = 4);
         nir_def *w = nir_load_shared(&b, 1, 32,
            nir_iadd_imm(&b, addr, layout.plane_stride), .align_mul = 4);
         nir_def *lerp = nir_ffma(&b, z, nir_fsub(&b, nir_imm_float(&b, 1.0f), c),
                                  nir_fmul(&b, w, c));
         nir_store_shared(&b, nir_bcsel(&b, nir_fneu(&b, c, nir_imm_float(&b, 0.0f)), lerp, z),
                          addr, .write_mask = 1, .align_mul = 4);
         nir_pop_if(&b, NULL);
      }
      b.fp_math_ctrl = fp;
      nir_barrier(&b, .execution_scope = SCOPE_SUBGROUP, .memory_scope = SCOPE_SUBGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   }
   nir_def *primitive_words=nir_udiv_imm(&b,
      nir_iadd_imm(&b,nir_imul_imm(&b,nir_u2u32(&b,p),prim_vertices(s)),3),4);
   for (unsigned word=0;word<DIV_ROUND_UP(prim_vertices(s)*s->info.mesh.max_primitives_out,4);word+=32) {
      nir_push_if(&b,nir_ult(&b,nir_iadd_imm(&b,lane,word),primitive_words));
      /* Native64/256 helper increments high16 target word by32 and
       * shared-byte address by128, while each active lane emits one word. */
      nir_store_mesh_uvb_agx(&b,nir_imm_ivec2(&b,word<<16,
         layout.output_base+4*word),.flags=1);
      nir_pop_if(&b,NULL);
   }
   /* Plane k publishes UVS word k unless the fragment interpolation order is
    * a draw-time choice; then user planes take the root UVS index of their
    * slot, like an unlinked vertex shader's store_uvs_agx. */
   nir_def *selector[M5_MESH_MAX_PLANES] = {NULL};
   if (varyings->dynamic) {
      for (unsigned slot = 0; slot < 64; ++slot) {
         if (!varyings->components[slot])
            continue;
         nir_def *base = nir_u2u32(&b, nir_load_uvs_index_agx(&b,
            .io_semantics = {.location = slot, .num_slots = 1}));
         for (unsigned c = 0; c < varyings->components[slot]; ++c)
            selector[varyings->first[slot] + c] = nir_iadd_imm(&b, base, c);
      }
   }
   for(unsigned vertex=0;vertex<layout.export_vertices;vertex+=32) {
      nir_push_if(&b,nir_ult(&b,nir_iadd_imm(&b,lane,vertex),nir_u2u32(&b,v)));
      for(unsigned plane=0;plane<layout.planes;plane++) {
         const unsigned address=layout.position+layout.plane_stride*plane+4*vertex;
         /* Native mesh64 stores0x200000|component in round2. The
          * descriptor shared address names the batch base, as native does. */
         nir_def *descriptor = selector[plane]
            ? nir_vec2(&b, nir_ior_imm(&b, selector[plane], vertex << 16),
                       nir_imm_int(&b, address))
            : nir_imm_ivec2(&b, (vertex << 16) | plane, address);
         nir_store_mesh_uvb_agx(&b,descriptor,.flags=0);
      }
      nir_pop_if(&b,NULL);
   }
   nir_pop_if(&b,NULL);
   s->info.shared_size=layout.output_base+layout.bytes;
   s->info.outputs_written=0;s->info.per_primitive_outputs=0;
   /* The native output limits describe the exported vertices. */
   s->info.mesh.max_vertices_out=layout.export_vertices;
   nir_progress(true,impl,nir_metadata_none);
   if (varyings->cull_primitive || layout.provoking)
      NIR_PASS(_, s, nir_lower_vars_to_ssa);
   return true;
}

/* agx_nir_lower_m5_mesh gives every primitive private vertices and stores
 * its per-primitive user, layer and viewport words on all of them, so the
 * fragment shader reads them as ordinary flat inputs. Like the mesh side,
 * UVS words are 32-bit; narrower inputs are narrowed after the load. */
static bool
lower_m5_primitive_input(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   (void)data;
   if (i->intrinsic != nir_intrinsic_load_per_primitive_input)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(i);
   const unsigned location = sem.location;
   if (!((location >= VARYING_SLOT_VAR0 && location < VARYING_SLOT_VAR0 + 32) ||
         location == VARYING_SLOT_LAYER || location == VARYING_SLOT_VIEWPORT ||
         location == VARYING_SLOT_PRIMITIVE_ID) ||
       (i->def.bit_size != 32 && i->def.bit_size != 16))
      return false;
   b->cursor = nir_before_instr(&i->instr);
   const nir_alu_type base =
      nir_alu_type_get_base_type(nir_intrinsic_dest_type(i));
   nir_def *value = nir_load_input(b, i->def.num_components, 32,
      i->src[0].ssa, .base = nir_intrinsic_base(i),
      .component = nir_intrinsic_component(i), .dest_type = base | 32,
      .io_semantics = sem);
   if (i->def.bit_size != 32)
      value = nir_convert_to_bit_size(b, value, base, i->def.bit_size);
   b->shader->info.per_primitive_inputs &=
      ~BITFIELD64_RANGE(location, MAX2(sem.num_slots, 1));
   nir_def_replace(&i->def, value);
   return true;
}

bool
agx_nir_lower_m5_mesh_fragment(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   return nir_shader_intrinsics_pass(s, lower_m5_primitive_input,
                                     nir_metadata_control_flow, NULL);
}

/* The native object and matching mesh both access the same pointer:
 * normalized SR62/W32 + SR63/W16. */
static bool
lower_payload(nir_builder *b,nir_intrinsic_instr *i,void *data)
{
   const unsigned payload_base = *(const unsigned *)data;
   if (i->intrinsic==nir_intrinsic_barrier &&
       (nir_intrinsic_memory_modes(i)&nir_var_mem_task_payload)) {
      nir_intrinsic_set_memory_modes(i,
         (nir_intrinsic_memory_modes(i)&~nir_var_mem_task_payload)|nir_var_mem_global);
      return true;
   }
   const bool atomic=i->intrinsic==nir_intrinsic_task_payload_atomic ||
                     i->intrinsic==nir_intrinsic_task_payload_atomic_swap;
   bool store=i->intrinsic==nir_intrinsic_store_task_payload;
   if (!store && !atomic && i->intrinsic!=nir_intrinsic_load_task_payload)
      return false;
   b->cursor=nir_before_instr(&i->instr);
   nir_def *pointer=nir_pack_64_2x32(b,nir_load_mesh_payload_address_agx(b));
   nir_def *offset=i->src[store ? 1 : 0].ssa;
   nir_def *addr=nir_iadd(b,pointer,nir_u2u64(b,
                            nir_iadd_imm(b,offset,nir_intrinsic_base(i)+payload_base)));
   if (atomic) {
      /* The payload is ordinary memory at the payload address: its atomics
       * are global atomics there, lowered further like any other. */
      const nir_atomic_op op=nir_intrinsic_atomic_op(i);
      nir_def *r=i->intrinsic==nir_intrinsic_task_payload_atomic_swap
         ? nir_global_atomic_swap(b,i->def.bit_size,addr,i->src[1].ssa,
                                  i->src[2].ssa,.atomic_op=op)
         : nir_global_atomic(b,i->def.bit_size,addr,i->src[1].ssa,
                             .atomic_op=op);
      nir_def_replace(&i->def,r);
   } else if (store) {
      nir_store_global(b,i->src[0].ssa,addr,
         .write_mask=nir_intrinsic_write_mask(i),
         .align_mul=nir_intrinsic_align_mul(i),
         .align_offset=nir_intrinsic_align_offset(i),
         .access=nir_intrinsic_access(i));
      nir_instr_remove(&i->instr);
   } else {
      nir_def_replace(&i->def,nir_load_global(b,i->def.num_components,
         i->def.bit_size,addr,.align_mul=nir_intrinsic_align_mul(i),
         .align_offset=nir_intrinsic_align_offset(i),
         .access=nir_intrinsic_access(i)));
   }
   return true;
}

bool
agx_nir_lower_m5_task_payload(nir_shader *s, unsigned max_payload_B,
                              unsigned payload_base_B)
{
   if (s->info.task_payload_size>max_payload_B) return false;
   bool reads=false;
   nir_foreach_function_impl(impl,s) {
      nir_foreach_block(block,impl) {
         nir_foreach_instr(in,block) {
            if (in->type!=nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
            reads |= i->intrinsic==nir_intrinsic_load_task_payload ||
                     i->intrinsic==nir_intrinsic_load_mesh_payload_address_agx;
            reads |= i->intrinsic==nir_intrinsic_task_payload_atomic ||
                     i->intrinsic==nir_intrinsic_task_payload_atomic_swap;
            if (i->intrinsic==nir_intrinsic_launch_mesh_workgroups_with_payload_deref)
               return false;
         }
      }
   }
   if (reads && s->info.stage==MESA_SHADER_MESH) {
      nir_builder b=nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(s)));
      nir_mesh_payload_entry_barrier_agx(&b);
   }
   nir_shader_intrinsics_pass(s,lower_payload,nir_metadata_control_flow,
                              &payload_base_B);
   return true;
}

/* Native object->mesh record at the payload address (SR62/SR63): native G17G
 * witnesses (lane-mesh nwit2 otpg/mtpg) store the child grid
 * (threadgroups_per_grid) at +0 and the payload from +16; the mesh reads its
 * threadgroups_per_grid from +0 after the payload-entry barrier.
 *
 * Only a mesh that reads threadgroups_per_grid needs that record. For any
 * other mesh the object writes no record and both stages put the payload at
 * +0. With the record, a full 1 KiB payload measured as lost workgroups of
 * the last children (those reading its last words), consistent with the
 * next task's record landing on them. Shared with hk_shader.c and
 * agx_vdm_mesh_core.h via AGX_M5_TASK_RECORD_PAYLOAD_BYTES. */

static bool
lower_mesh_record_grid(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   (void)data;
   if (i->intrinsic != nir_intrinsic_load_num_workgroups)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *pointer = nir_pack_64_2x32(b, nir_load_mesh_payload_address_agx(b));
   nir_def *grid = nir_load_global(b, 3, 32, pointer, .align_mul = 16);
   nir_def_replace(&i->def, nir_u2uN(b, grid, i->def.bit_size));
   return true;
}

bool
agx_nir_lower_m5_mesh_task_record(nir_shader *s)
{
   if (s->info.stage != MESA_SHADER_MESH)
      return false;
   return nir_shader_intrinsics_pass(s, lower_mesh_record_grid,
                                     nir_metadata_control_flow, NULL);
}

static bool
lower_launch(nir_builder *b,nir_intrinsic_instr *i,void *data)
{
   if (i->intrinsic!=nir_intrinsic_launch_mesh_workgroups) return false;
   const unsigned grid=*(unsigned *)data;
   b->cursor=nir_before_instr(&i->instr);
   nir_push_if(b,nir_ieq_imm(b,nir_load_local_invocation_index(b),0));
   nir_store_shared(b,i->src[0].ssa,nir_imm_int(b,grid),
                       .write_mask=7,.align_mul=4);
   nir_pop_if(b,NULL);
   nir_instr_remove(&i->instr);
   return true;
}

/* The API task (object) stage. The child grid is published with UVSW
 * from lanes 0..2 of SIMD group 0. Payload-carrying Apple object shaders
 * store the EmitMeshTasks grid in threadgroup memory, rendezvous and pass
 * the object payload-exit barrier first (_agc.object.write_childcount);
 * payload-free objects take the grid of invocation 0 from registers, like
 * the mesh-only passthrough object. Workgroups of fewer than three
 * invocations run on three physical lanes with the API body under its exact
 * invocation mask. Payload accesses use the native SR62/SR63 payload
 * address; callers bound max_payload_B to what the running kernel backs
 * (zero refuses every payload access). grid_record also stores the child grid
 * at +0 of the payload address and moves the payload to +16, for a mesh that
 * reads its threadgroups_per_grid (agx_nir_lower_m5_mesh_task_record); the
 * mesh must then address the payload from +16 too, and from +0 without it.
 */
bool
agx_nir_lower_m5_task(nir_shader *s, unsigned max_payload_B, bool grid_record)
{
   const unsigned threads = s->info.workgroup_size[0] *
                            s->info.workgroup_size[1] *
                            s->info.workgroup_size[2];
   if (s->info.stage != MESA_SHADER_TASK || s->info.workgroup_size_variable ||
       threads == 0 || threads > 1024)
      return false;
   const unsigned physical_threads = MAX2(threads, 3);
   unsigned launches = 0;
   bool payload_access = false;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            switch (i->intrinsic) {
            case nir_intrinsic_launch_mesh_workgroups:
               launches++;
               break;
            case nir_intrinsic_load_task_payload:
            case nir_intrinsic_store_task_payload:
            case nir_intrinsic_task_payload_atomic:
            case nir_intrinsic_task_payload_atomic_swap:
               if (!max_payload_B)
                  return false;
               payload_access = true;
               break;
            case nir_intrinsic_barrier:
               /* As for padded mesh workgroups. */
               if (physical_threads != threads &&
                   !m5_padded_barrier_ok(block, i))
                  return false;
               break;
            default:
               break;
            }
         }
      }
   }
   const unsigned grid = ALIGN_POT(s->info.shared_size, 16);
   /* A body that uses all 32 KiB of threadgroup memory leaves no room for
    * the grid words; they then alias word 0 once the body is done. */
   const bool alias_grid = grid + 16 > 32768;
   if (!launches || s->info.shared_size > 32768 ||
       !agx_nir_lower_m5_task_payload(
          s, max_payload_B,
          grid_record ? AGX_M5_TASK_RECORD_PAYLOAD_BYTES : 0))
      return false;
   /* SPIR-V EmitMeshTasksEXT terminates the invocation: vtn follows the
    * launch with a halt, which would skip the child-count epilogue appended
    * below (EmitMeshTasks inside an if/else left both branches halting and
    * published undefined counts). A task shader has no other halt source, so
    * the launch ends main instead. */
   {
      bool halts = false;
      nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
         nir_foreach_instr(in, block) {
            if (in->type == nir_instr_type_jump &&
                nir_instr_as_jump(in)->type == nir_jump_halt) {
               nir_instr_as_jump(in)->type = nir_jump_return;
               halts = true;
            }
         }
      }
      if (halts)
         NIR_PASS(_, s, nir_lower_returns);
   }
   /* A payload-free object publishes the grid of invocation 0 from
    * registers, like Apple's mesh-only passthrough object: no threadgroup
    * memory and no rendezvous. Only payload-carrying objects need the
    * threadgroup grid, the rendezvous and the payload-exit barrier of
    * _agc.object.write_childcount. (The latter shape faulted once on R239
    * with no 0x1ca48 mesh rendering buffer bound; it stays behind the
    * payload limit.) The direct form keys off that limit, not off the grid
    * record: a payload-free object on a kernel that backs the payload keeps
    * the rendezvous form, record or not. */
   const bool direct = !max_payload_B && !s->info.task_payload_size &&
                       !payload_access;
   nir_function_impl *entry = nir_shader_get_entrypoint(s);
   nir_variable *grid_var = NULL;
   if (direct) {
      grid_var = nir_local_variable_create(entry, glsl_uvec_type(3), "task_grid");
      nir_foreach_block(block, entry) {
         nir_foreach_instr_safe(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            if (i->intrinsic != nir_intrinsic_launch_mesh_workgroups)
               continue;
            nir_builder b = nir_builder_at(nir_before_instr(in));
            nir_store_var(&b, grid_var, i->src[0].ssa, 7);
            nir_instr_remove(in);
         }
      }
      nir_builder b = nir_builder_at(nir_before_impl(entry));
      nir_store_var(&b, grid_var, nir_imm_ivec3(&b, 0, 0, 0), 7);
   } else {
      if (grid_record) {
         /* Native object shaders also publish the child grid at +0 of the
          * record for the mesh stage's threadgroups_per_grid. */
         nir_foreach_block(block, entry) {
            nir_foreach_instr(in, block) {
               if (in->type != nir_instr_type_intrinsic ||
                   nir_instr_as_intrinsic(in)->intrinsic !=
                      nir_intrinsic_launch_mesh_workgroups)
                  continue;
               nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
               nir_builder b = nir_builder_at(nir_before_instr(in));
               nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
               nir_store_global(&b, i->src[0].ssa,
                  nir_pack_64_2x32(&b, nir_load_mesh_payload_address_agx(&b)),
                  .write_mask = 7, .align_mul = 16);
               nir_pop_if(&b, NULL);
            }
         }
      }
      if (alias_grid) {
         grid_var = nir_local_variable_create(entry, glsl_uvec_type(3), "task_grid");
         nir_foreach_block(block, entry) {
            nir_foreach_instr_safe(in, block) {
               if (in->type != nir_instr_type_intrinsic ||
                   nir_instr_as_intrinsic(in)->intrinsic !=
                      nir_intrinsic_launch_mesh_workgroups)
                  continue;
               nir_builder b = nir_builder_at(nir_before_instr(in));
               nir_store_var(&b, grid_var, nir_instr_as_intrinsic(in)->src[0].ssa, 7);
               nir_instr_remove(in);
            }
         }
         nir_builder b = nir_builder_at(nir_before_impl(entry));
         nir_store_var(&b, grid_var, nir_imm_ivec3(&b, 0, 0, 0), 7);
      } else {
         unsigned grid_offset = grid;
         nir_shader_intrinsics_pass(s, lower_launch, nir_metadata_none, &grid_offset);
      }
   }
   if (physical_threads != threads) {
      nir_foreach_block(block, entry) {
         nir_foreach_instr_safe(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            nir_builder b = nir_builder_at(nir_before_instr(in));
            nir_def *replacement = NULL;
            if (i->intrinsic == nir_intrinsic_load_workgroup_size) {
               replacement = nir_imm_ivec3(&b, s->info.workgroup_size[0],
                  s->info.workgroup_size[1], s->info.workgroup_size[2]);
            } else if (i->intrinsic == nir_intrinsic_load_local_invocation_id ||
                       i->intrinsic == nir_intrinsic_load_global_invocation_id) {
               nir_def *index = nir_load_local_invocation_index(&b);
               const unsigned x = s->info.workgroup_size[0];
               const unsigned y = s->info.workgroup_size[1];
               replacement = nir_vec3(&b, nir_umod_imm(&b, index, x),
                  nir_umod_imm(&b, nir_udiv_imm(&b, index, x), y),
                  nir_udiv_imm(&b, index, x * y));
               if (i->intrinsic == nir_intrinsic_load_global_invocation_id) {
                  nir_def *size = nir_imm_ivec3(&b, s->info.workgroup_size[0],
                     s->info.workgroup_size[1], s->info.workgroup_size[2]);
                  replacement = nir_iadd(&b,
                     nir_imul(&b, nir_load_workgroup_id(&b), size), replacement);
                  replacement = nir_u2uN(&b, replacement, i->def.bit_size);
               }
            }
            if (replacement) {
               nir_def_rewrite_uses(&i->def, replacement);
               nir_instr_remove(in);
            }
         }
      }
      m5_mask_padded_body(entry, threads);
      s->info.workgroup_size[0] = physical_threads;
      s->info.workgroup_size[1] = s->info.workgroup_size[2] = 1;
   }
   nir_builder b = nir_builder_at(nir_after_cf_list(&entry->body));
   if (direct) {
      nir_def *value = nir_read_invocation(&b, nir_load_var(&b, grid_var),
                                           nir_imm_int(&b, 0));
      nir_def *lane = nir_load_subgroup_invocation(&b);
      nir_push_if(&b, nir_iand(&b, nir_ieq_imm(&b, nir_load_subgroup_id(&b), 0),
                               nir_ult_imm(&b, lane, 3)));
      nir_def *count = nir_bcsel(&b, nir_ieq_imm(&b, lane, 0),
         nir_channel(&b, value, 0),
         nir_bcsel(&b, nir_ieq_imm(&b, lane, 1), nir_channel(&b, value, 1),
                   nir_channel(&b, value, 2)));
      nir_store_object_childcount_agx(&b, count);
      nir_pop_if(&b, NULL);
      /* Object threadgroup memory is allocated in 64-byte units. */
      if (s->info.shared_size)
         s->info.shared_size = ALIGN_POT(s->info.shared_size, 64);
      nir_progress(true, entry, nir_metadata_none);
      NIR_PASS(_, s, nir_lower_vars_to_ssa);
      return true;
   }
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
   if (alias_grid) {
      /* Every invocation has finished with the body's shared memory. */
      nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
      nir_store_shared(&b, nir_load_var(&b, grid_var), nir_imm_int(&b, 0),
                       .write_mask = 7, .align_mul = 4);
      nir_pop_if(&b, NULL);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                  .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_mem_shared);
   }
   nir_object_payload_exit_barrier_agx(&b);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_push_if(&b, nir_ult_imm(&b, lane, 3));
   nir_def *value = nir_load_shared(&b, 1, 32,
      nir_iadd_imm(&b, nir_imul_imm(&b, lane, 4), alias_grid ? 0 : grid),
      .align_mul = 4);
   nir_store_object_childcount_agx(&b, value);
   nir_pop_if(&b, NULL);
   s->info.shared_size = alias_grid ? ALIGN_POT(s->info.shared_size, 64)
                                    : ALIGN_POT(grid + 16, 64);
   nir_progress(true, entry, nir_metadata_none);
   if (alias_grid)
      NIR_PASS(_, s, nir_lower_vars_to_ssa);
   return true;
}
