/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef HK_AGX3_AS_FORMAT_H
#define HK_AGX3_AS_FORMAT_H

#include "asahi/lib/agx_chip.h"

/*
 * Native acceleration-structure formats. The device's chip selects one, and
 * that format's encoder writes the records that its own refit, traversal,
 * serialization, relocation and copies read:
 *
 *  - GEN1: G17P and the other native AGX3 chips. bvh/agx3_gen1_*.comp encode
 *    four node jobs per 32-lane workgroup through an indirect dispatch
 *    written after the plan state, and may pair triangles sharing an edge in
 *    one record; every child has a 64-byte record slot; images are aligned
 *    to 16 KiB (hk_acceleration_structure.c, libagx/gen1_as.h,
 *    compiler/agx3_rt_gen1_*).
 *  - M5: G17G and G17S. bvh/agx3_*.comp encode one node per 8-lane
 *    workgroup, batched through a dense job map, with 48-byte bounding-box
 *    records; images are aligned to 1 KiB (hk_m5_acceleration_structure.c).
 */
enum hk_agx3_as_format {
   HK_AGX3_AS_FORMAT_GEN1,
   HK_AGX3_AS_FORMAT_M5,
};

static inline enum hk_agx3_as_format
hk_agx3_as_format_for_chip(enum agx_chip chip)
{
   return agx_chip_g17_m5(chip) ? HK_AGX3_AS_FORMAT_M5 : HK_AGX3_AS_FORMAT_GEN1;
}

/* Bytes of a native child record slot, common to both formats. */
#define HK_AGX3_AS_SLOT_B 64u

/* Image alignment. */
#define HK_AGX3_GEN1_IMAGE_ALIGN 16384u
#define HK_AGX3_M5_IMAGE_ALIGN   1024u

/* Encoding scratch after hk_agx3_plan_state's offset: GEN1 keeps the node
 * encoding dispatch arguments (HK_AGX3_PLAN_ENCODE_DISPATCH) there, M5 only
 * the plan state. */
#define HK_AGX3_GEN1_PLAN_TAIL_B 48u
#define HK_AGX3_M5_PLAN_TAIL_B   32u

/* Bounding-box leaf record size. The M5 record size,
 * HK_AGX3_M5_AABB_RECORD_B, is defined with its layout in
 * bvh/hk_build_interface.h, which the encoder programs share. */
#define HK_AGX3_GEN1_AABB_RECORD_B 64u

/* Lanes of one node-encoding workgroup. */
#define HK_AGX3_GEN1_NODE_GROUP_LANES 32u
#define HK_AGX3_M5_NODE_GROUP_LANES   8u

static inline unsigned
hk_agx3_as_image_align(enum hk_agx3_as_format format)
{
   return format == HK_AGX3_AS_FORMAT_M5 ? HK_AGX3_M5_IMAGE_ALIGN
                                         : HK_AGX3_GEN1_IMAGE_ALIGN;
}

static inline unsigned
hk_agx3_as_plan_tail(enum hk_agx3_as_format format)
{
   return format == HK_AGX3_AS_FORMAT_M5 ? HK_AGX3_M5_PLAN_TAIL_B
                                         : HK_AGX3_GEN1_PLAN_TAIL_B;
}

#endif
