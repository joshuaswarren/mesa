/* SPDX-License-Identifier: MIT */
#ifndef AGX_RT_BLAS_GEN3_H
#define AGX_RT_BLAS_GEN3_H

#include "agx_rt_bvh.h"

/* Narrow Gen3 BLAS serializer for one unindexed opaque triangle in geometry 0;
 * unassigned descriptor trailer words are kept literal. */
#define AGX_RT_BLAS_GEN3_SINGLE_TRIANGLE_SIZE 504u

/* Write the Gen3 single-geometry triangle envelope (header and 184-byte
 * trailing descriptor) around traversal graph slots [3, end_slot).
 * `opaque_build_word` is placed at offset 0xb8; degenerate bounds fail. */
static inline bool
agx_rt_blas_gen3_envelope(void *packed, size_t available, uint64_t gpu_va,
                         uint32_t end_slot, uint32_t primitive_count,
                         uint32_t node_count, uint32_t depth,
                         const float minimum[3], const float maximum[3],
                         uint32_t opaque_build_word)
{
   const uint64_t tail = (uint64_t)end_slot * 64;
   const uint64_t size = tail + 184;
   if (!packed || !minimum || !maximum || !gpu_va || (gpu_va & 0x3ff) ||
       gpu_va >= (UINT64_C(1) << 48) ||
       size > (UINT64_C(1) << 48) - gpu_va || end_slot < 3 || size > available ||
       !primitive_count || !node_count || !depth)
      return false;
   float extent[3];
   for (unsigned a = 0; a < 3; ++a) {
      if (!agx_rt_bvh_is_finite(minimum[a]) ||
          !agx_rt_bvh_is_finite(maximum[a]) || minimum[a] > maximum[a])
         return false;
      extent[a] = maximum[a] - minimum[a];
   }
   const float xy = extent[0] * extent[1];
   const float xz = extent[0] * extent[2];
   const float yz = extent[1] * extent[2];
   const float area = (xy + xz) + yz;
   if (!(area > 0) || !agx_rt_bvh_is_finite(area))
      return false;
   const float inverse = 1.0f / area;
   if (!agx_rt_bvh_is_finite(inverse))
      return false;

   uint8_t header[128] = {0};
   agx_store_le64(header, UINT64_C(0x080000002300f032));
   agx_store_le64(header + 8, size);
   for (unsigned a = 0; a < 3; ++a) {
      agx_store_le32(header + 16 + a * 4, agx_rt_bvh_bitcast_f32_to_u32(minimum[a]));
      agx_store_le32(header + 28 + a * 4, agx_rt_bvh_bitcast_f32_to_u32(maximum[a]));
   }
   for (unsigned off = 40; off <= 88; off += 8)
      agx_store_le64(header + off, off == 80 ? gpu_va : tail);
   agx_store_le32(header + 96, 1);
   agx_store_le32(header + 112, 1);
   agx_store_le32(header + 120, opaque_build_word);
   agx_store_le32(header + 124, 0x01000000);

   uint32_t descriptor[46] = {
      primitive_count, primitive_count, 0, 0, node_count, depth, 1, 0,
      1, 0, 0, 0, 0, 0x3f800000, 0, 0, 2, 1, 0, 0, 2, 10, 0, 0,
      0, 0, 0, 0, 0x01000000, 0x01000000, 0x00000101, 0x00010000,
      0x00010000, 0x00ff0001, 0x02000000, 0x01000005,
      0x03000000, 1, 0, 0x3f800000, 0, 0, 0, 0, 0, 0,
   };
   descriptor[27] = agx_rt_bvh_bitcast_f32_to_u32(inverse);
   uint8_t *bytes = packed;
   memcpy(bytes + 64, header, sizeof(header));
   for (unsigned i = 0; i < 46; ++i)
      agx_store_le32(bytes + tail + i * 4, descriptor[i]);
   return true;
}

/* Emit one native root and its contiguous QTB/geometry child region. The
 * caller owns the allocation header and metadata between/after these regions.
 * One geometry, explicit sequential primitive IDs, 1..8 unindexed triangles.
 * `child_slot` is the first QTB's 64-byte slot relative to the AS base. */
static inline bool
agx_rt_blas_gen3_small_graph(void *root_out, void *children, size_t available,
                            const float (*vertices)[3][3], unsigned count,
                            uint32_t child_slot)
{
   if (!root_out || !children || !vertices || count < 1 || count > 8)
      return false;
   const unsigned capacity = count < 4 ? count : 4;
   const unsigned blocks = (count + capacity - 1) / capacity;
   const unsigned bytes = (blocks + count) * 64;
   if (available < bytes)
      return false;
   float minimum[3], maximum[3];
   for (unsigned a = 0; a < 3; ++a) {
      minimum[a] = maximum[a] = vertices[0][0][a];
      for (unsigned t = 0; t < count; ++t)
         for (unsigned v = 0; v < 3; ++v) {
            float x = vertices[t][v][a];
            if (!agx_rt_bvh_is_finite(x)) return false;
            if (x < minimum[a]) minimum[a] = x;
            if (x > maximum[a]) maximum[a] = x;
         }
   }
   struct agx_rt_bvh_gen1_quantization_frame frame;
   if (!agx_rt_bvh_gen1_quantization_frame_fit(minimum, maximum, &frame))
      return false;
   for (unsigned a = 0; a < 3; ++a)
      if (!frame.exponent[a]) frame.exponent[a] = 1;

   /* Native multi-triangle QTB and explicit-ID flags. Do not copy these
    * flags to the distinct single-triangle type-2 representation. */
   struct agx_rt_bvh_gen1_header root = {
      .words = {0xa0, 0, 0, 0, 0, 0, 0, 0x40c0}};
   if (!agx_rt_bvh_gen1_header_set_child_layout(
          &root, 0, count, child_slot, 8, capacity, false) ||
       !agx_rt_bvh_gen1_quantization_frame_pack(&root, &frame))
      return false;

   uint8_t data[640] = {0};
   for (unsigned b = 0; b < blocks; ++b) {
      unsigned n = count - b * capacity;
      if (n > capacity) n = capacity;
      if (!agx_rt_qtb_gen3_type6_from_triangles(
             data + b * 64, 64, vertices + b * capacity, n))
         return false;
   }
   for (unsigned t = 0; t < count; ++t) {
      float lo[3], hi[3];
      uint8_t *geometry = data + (blocks + t) * 64;
      agx_store_le32(geometry, 0x48);
      for (unsigned a = 0; a < 3; ++a) {
         lo[a] = hi[a] = vertices[t][0][a];
         for (unsigned v = 0; v < 4; ++v) {
            float x = vertices[t][v < 3 ? v : 2][a];
            agx_store_le32(geometry + 4 + a * 16 + v * 4,
                          agx_rt_bvh_bitcast_f32_to_u32(x));
            if (x < lo[a]) lo[a] = x;
            if (x > hi[a]) hi[a] = x;
         }
      }
      agx_store_le32(geometry + 56, t);
      agx_store_le32(geometry + 60, t);
      if (!agx_rt_bvh_gen1_header_set_leaf_lane_end(&root, t, t + 1) ||
          !agx_rt_bvh_gen1_header_encode_child_bounds(&root, t, lo, hi))
         return false;
   }
   uint8_t packed_root[64];
   if (!agx_rt_bvh_gen1_header_pack(packed_root, 64, &root)) return false;
   memcpy(root_out, packed_root, 64);
   memcpy(children, data, bytes);
   return true;
}

static inline bool
agx_rt_blas_gen3_single_triangle(void *packed, size_t available, uint64_t gpu_va,
                                const float vertices[3][3])
{
   if (!packed || !vertices || available < AGX_RT_BLAS_GEN3_SINGLE_TRIANGLE_SIZE ||
       !gpu_va || (gpu_va & 0x3ff) || gpu_va >= (UINT64_C(1) << 48))
      return false;

   float minimum[3], maximum[3];
   for (unsigned a = 0; a < 3; ++a) {
      minimum[a] = maximum[a] = vertices[0][a];
      for (unsigned v = 0; v < 3; ++v) {
         if (!agx_rt_bvh_is_finite(vertices[v][a]))
            return false;
         if (vertices[v][a] < minimum[a]) minimum[a] = vertices[v][a];
         if (vertices[v][a] > maximum[a]) maximum[a] = vertices[v][a];
      }
   }

   struct agx_rt_bvh_gen1_quantization_frame frame;
   if (!agx_rt_bvh_gen1_quantization_frame_fit(minimum, maximum, &frame))
      return false;
   for (unsigned a = 0; a < 3; ++a)
      if (!frame.exponent[a]) frame.exponent[a] = 1;

   /* Preserve the observed fixed root bits; all address/count/bound fields
    * below are emitted through the existing independently checked codecs. */
   struct agx_rt_bvh_gen1_header root = {.words = {0x80, 0, 0, 0, 0, 0, 0, 0xc0}};
   if (!agx_rt_bvh_gen1_header_set_child_layout(&root, 0, 1, 3, 8, 1, false) ||
       !agx_rt_bvh_gen1_header_set_leaf_lane_end(&root, 0, 1) ||
       !agx_rt_bvh_gen1_quantization_frame_pack(&root, &frame) ||
       !agx_rt_bvh_gen1_header_encode_child_bounds(&root, 0, minimum, maximum))
      return false;

   uint8_t result[AGX_RT_BLAS_GEN3_SINGLE_TRIANGLE_SIZE] = {0};
   agx_rt_bvh_gen1_header_pack(result, sizeof(result), &root);
   agx_store_le64(result + 0x40, UINT64_C(0x080000000300f032));
   agx_store_le64(result + 0x48, sizeof(result));
   for (unsigned a = 0; a < 3; ++a) {
      agx_store_le32(result + 0x50 + a * 4, agx_rt_bvh_bitcast_f32_to_u32(minimum[a]));
      agx_store_le32(result + 0x5c + a * 4, agx_rt_bvh_bitcast_f32_to_u32(maximum[a]));
   }
   for (unsigned off = 0x68; off <= 0x98; off += 8)
      agx_store_le64(result + off, off == 0x90 ? gpu_va : 0x140);
   agx_store_le32(result + 0xa0, 1);
   agx_store_le32(result + 0xb0, 1);
   agx_store_le32(result + 0xb8, 0x3ff33334);
   agx_store_le32(result + 0xbc, 0x01000000);
   if (!agx_rt_qtb_gen3_type2_from_triangle(result + 0xc0, 64, vertices))
      return false;

   /* Compact unindexed geometry record. Unlike the older full-data fixture,
    * this native variant starts with 0x08, not the 0x50 full-data marker. */
   agx_store_le32(result + 0x100, 8);
   for (unsigned a = 0; a < 3; ++a)
      for (unsigned v = 0; v < 3; ++v)
         agx_store_le32(result + 0x104 + 16 * a + 4 * v,
                       agx_rt_bvh_bitcast_f32_to_u32(vertices[v][a]));

   static const uint32_t descriptor[46] = {
      1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0,
      0, 0x3f800000, 0, 0, 0, 1, 0, 0, 0, 10, 0, 0,
      0, 0, 0, 0x3d800000, 0x01000000, 0x01000000, 0x00000101, 0x00010000,
      0x00010000, 0x00ff0001, 0x02000000, 0x01000005,
      0x03000000, 1, 0, 0x3f800000, 0, 0, 0, 0, 0, 0,
   };
   for (unsigned i = 0; i < 46; ++i)
      agx_store_le32(result + 0x140 + i * 4, descriptor[i]);
   memcpy(packed, result, sizeof(result));
   return true;
}
#endif
