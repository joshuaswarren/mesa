/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include <stdalign.h>
#include <stdio.h>
#include <string.h>
#include "agx_ppp.h"

/* agx_ppp_push_merged() packs a record into a stack scratch and ORs it with a
 * precomputed record a word at a time (agx_ppp_push_merged_blobs(), which
 * asserts 4-byte alignment). Where the compiler places a byte array is not
 * deterministic, so the scratch's element type is pinned here as well as its
 * size, its address and the merged result. */
#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

#define CHECK_SCRATCH(T)                                                       \
   do {                                                                        \
      CHECK(alignof(__typeof__(_tmp[0])) >= 4);                                \
      CHECK(sizeof(_tmp) == AGX_##T##_LENGTH);                                 \
      CHECK(((uintptr_t)_tmp & 3) == 0);                                       \
   } while (0)

int
main(void)
{
   alignas(4) uint8_t buf[64];
   memset(buf, 0xcc, sizeof(buf));
   struct agx_ppp_update ppp = {
      .head = buf,
      .cpu_base = buf,
      .total_size = 3 * 4,
   };

   /* FRAGMENT_CONTROL: the precomputed word sets the pass type, the push
    * adds tag_write_disable (hk_cmd_draw.c, agx_state.c). */
   struct agx_fragment_control_packed control, control_all;
   agx_pack(&control, FRAGMENT_CONTROL, cfg) {
      cfg.pass_type = AGX_PASS_TYPE_PUNCH_THROUGH;
      cfg.scissor_enable = true;
   }
   agx_pack(&control_all, FRAGMENT_CONTROL, cfg) {
      cfg.pass_type = AGX_PASS_TYPE_PUNCH_THROUGH;
      cfg.scissor_enable = true;
      cfg.tag_write_disable = true;
   }
   agx_ppp_push_merged(&ppp, FRAGMENT_CONTROL, cfg, control) {
      CHECK_SCRATCH(FRAGMENT_CONTROL);
      cfg.tag_write_disable = true;
   }
   CHECK(ppp.head == buf + 4);
   CHECK(!memcmp(buf, &control_all, 4));

   /* FRAGMENT_FACE: the precomputed word holds depth state, the push adds the
    * per-face stencil reference. */
   struct agx_fragment_face_packed face, face_all;
   agx_pack(&face, FRAGMENT_FACE, cfg) {
      cfg.line_width = 0x1f;
      cfg.polygon_mode = AGX_POLYGON_MODE_LINE;
      cfg.disable_depth_write = true;
      cfg.depth_function = AGX_ZS_FUNC_GEQUAL;
   }
   agx_pack(&face_all, FRAGMENT_FACE, cfg) {
      cfg.stencil_reference = 0xa5;
      cfg.line_width = 0x1f;
      cfg.polygon_mode = AGX_POLYGON_MODE_LINE;
      cfg.disable_depth_write = true;
      cfg.depth_function = AGX_ZS_FUNC_GEQUAL;
   }
   agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, face) {
      CHECK_SCRATCH(FRAGMENT_FACE);
      cfg.stencil_reference = 0xa5;
   }
   CHECK(ppp.head == buf + 8);
   CHECK(!memcmp(buf + 4, &face_all, 4));

   /* CULL: the rasterizer state's word plus the draw's own bits. A field set
    * on both sides stays set (OR, not replace). */
   struct agx_cull_packed cull, cull_all;
   agx_pack(&cull, CULL, cfg) {
      cfg.cull_back = true;
      cfg.depth_clip = true;
      cfg.front_face_ccw = true;
   }
   agx_pack(&cull_all, CULL, cfg) {
      cfg.cull_back = true;
      cfg.depth_clip = true;
      cfg.front_face_ccw = true;
      cfg.rasterizer_discard = true;
   }
   agx_ppp_push_merged(&ppp, CULL, cfg, cull) {
      CHECK_SCRATCH(CULL);
      cfg.depth_clip = true;
      cfg.rasterizer_discard = true;
   }
   CHECK(ppp.head == buf + 12);
   CHECK(!memcmp(buf + 8, &cull_all, 4));

   /* Nothing past the update is written. */
   for (unsigned i = 12; i < sizeof(buf); ++i)
      CHECK(buf[i] == 0xcc);

   printf("ppp_push_merged: OK\n");
   return 0;
}
