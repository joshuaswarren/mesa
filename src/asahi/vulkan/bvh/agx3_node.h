/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Native G17P node encoding helpers shared by node encoding (agx3_leaf.comp)
 * and refit propagation (agx3_refit.comp).
 */
#ifndef HK_AGX3_NODE_H
#define HK_AGX3_NODE_H

bool
finite(float x)
{
   return !isnan(x) && !isinf(x);
}
float
origin_down(float x)
{
   uint bits = floatBitsToUint(x);
   uint upper_bits = bits >> 16;
   if ((bits & 0x80000000u) != 0 && (bits & 65535u) != 0)
      ++upper_bits;
   return uintBitsToFloat(upper_bits << 16);
}

float
step(uint exponent)
{
   return uintBitsToFloat(exponent << 23);
}

float
inv_step(uint exponent)
{
   /* Exact reciprocal of step(exponent) = 2^(exponent - 127) for 1 <= exponent <= 254. */
   float s = uintBitsToFloat((254u - min(exponent, 253u)) << 23);
   return exponent == 254u ? s * 0.5 : s;
}

bool
fit_root(vec3 lo, vec3 hi, out vec3 base, out uvec3 exponent)
{
   for (uint a = 0; a < 3; ++a) {
      base[a] = origin_down(lo[a]);
      precise float extent = hi[a] - base[a];
      if (!finite(base[a]) || !finite(extent))
         return false;
      precise float unit = extent * (1.0 / 127.0);
      uint bits = floatBitsToUint(unit);
      uint e = (bits >> 23) & 255u;
      if (e != 0 && (bits & 0x7fffffu) != 0)
         ++e;
      e = max(e, 1u);
      /* step(e) is an exact power of two, so 127.0 * step(e) is exact in f32
       * (or +inf for e >= 249, where any finite extent satisfies <= 127 * step(e)). */
      while (e < 255u && extent > 127.0 * step(e))
         ++e;
      if (e == 255u)
         return false;
      exponent[a] = e;
   }
   return true;
}

/* Stores go through these helpers: glslang's precise propagation runs out
 * of memory on buffer-reference stores written out in main(). */
void
store32(uint64_t address, uint value)
{
   DEREF(REF(uint32_t)(address)) = value;
}

void
store_word64(uint64_t address, uint64_t value)
{
   DEREF(REF(uint64_t)(address)) = value;
}

void
store128(uint64_t address, uvec4 value)
{
   DEREF(REF(uvec4)(address)) = value;
}

void
store_aabb(uint64_t address, vec3 lo, vec3 hi)
{
   DEREF(REF(vk_aabb)(address)).min = lo;
   DEREF(REF(vk_aabb)(address)).max = hi;
}

/* Bounds a refit leaves for a job that failed: its ancestors fail too. */
#define HK_AGX3_FAILED_BOUNDS vec3(uintBitsToFloat(0x7fc00000u))

/* Root word t of a node with `count` children: its header bits, then the
 * bounds of child t (lo, hi) quantized against the node's base and
 * exponent. Clears `ok` when the bounds do not fit. */
uint64_t
root_word(uint t, uint count, uint kind, uint triangle_type0, uint child_slot,
          uint slot_bias, vec3 base, uvec3 exponent, vec3 lo, vec3 hi, inout bool ok)
{
   bool leaf = kind == HK_AGX3_NODE_LEAF, instances = kind == HK_AGX3_NODE_INSTANCES;
   bool aabbs = kind == HK_AGX3_NODE_AABBS;
   uint capacity = max(min(count, 4u), 1u);
   uint64_t word = uint64_t(0);
   if (t == 0)
      word = uint64_t((instances ? 0xe1u : 0xa0u) |
                      (leaf ? capacity << 1 : 0u) | (exponent.x << 8));
   else if (t <= 3)
      word = uint64_t(floatBitsToUint(base[t - 1u]) >> 16);
   else if (t == 4)
      word = uint64_t((exponent.y << 8) | exponent.z);
   else if (t == 5)
      word = uint64_t((child_slot + slot_bias) >> 16);
   else if (t == 6)
      word = uint64_t((child_slot + slot_bias) & 65535u);
   else
      word = instances ? uint64_t(0x1a00u | (count == 8u ? 0x3fu : count << 3))
             : aabbs ? uint64_t(0x18c0u | (count == 8u ? 0x3fu : count << 3))
             : leaf ? uint64_t(0x40c0u | (triangle_type0 << 8) |
                                       (count == 8u ? 0x3fu : count << 3))
                    : uint64_t(0x1f00u | (count == 8u ? 0u : count));
   if (t < count) {
      const uint shifts[6] = uint[6](57, 50, 43, 36, 29, 22);
      if (leaf || instances || aabbs)
         word |= uint64_t((t << 16) | (1u << 20));
      for (uint a = 0; a < 3; ++a) {
         float s = step(exponent[a]);
         float inv_s = inv_step(exponent[a]);
         uint qlo = uint(clamp(floor((lo[a] - base[a]) * inv_s), 0.0, 127.0));
         uint qhi = uint(clamp(ceil((hi[a] - base[a]) * inv_s), 0.0, 127.0));
         precise float rlo = base[a] + float(qlo) * s;
         precise float rhi = base[a] + float(qhi) * s;
         while (rlo > lo[a] && qlo > 0u) {
            --qlo;
            rlo = base[a] + float(qlo) * s;
         }
         while (rhi < hi[a] && qhi < 127u) {
            ++qhi;
            rhi = base[a] + float(qhi) * s;
         }
         if (rlo > lo[a] || rhi < hi[a])
            ok = false;
         word |= uint64_t(qlo) << shifts[a];
         word |= uint64_t(qhi) << shifts[a + 3];
      }
   }
   return word;
}

#endif /* HK_AGX3_NODE_H */
