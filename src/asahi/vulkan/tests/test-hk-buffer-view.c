/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Host unit tests for AGX2 and AGX3 texel buffer descriptors in
 * hk_CreateBufferView() and hk_texel_buffer_descriptor():
 *   - AGX2 texture.height and pbe.height include + tail_offset_el and reject
 *     views exceeding AGX_TEXTURE_BUFFER_MAX_HEIGHT cleanly.
 *   - AGX2 4-byte-aligned VK_FORMAT_R32G32B32_{SFLOAT,UINT,SINT} views keep a
 *     16-byte-aligned descriptor address and encode (offset & 0xf) / 4 in
 *     buffer_offset_sw, while rejecting non-4-byte-aligned offsets.
 *   - AGX3 (G17P) metadata, height, and RGB32 paths remain unchanged.
 */
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agx_nir_texture.h"
#include "agx_pack.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_pbe_agx3_core.h"
#include "hk_buffer.h"
#include "hk_buffer_view.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_instance.h"
#include "hk_native_descriptor.h"
#include "hk_native_descriptor_pack.h"
#include "hk_physical_device.h"
#include "vk_alloc.h"

struct test_fixture {
   struct hk_instance instance;
   struct hk_physical_device pdev;
   struct hk_device dev;
   struct hk_native_texture_descriptor *heap;
};

static void
fixture_init(struct test_fixture *f, enum agx_chip chip, uint32_t heap_entries)
{
   memset(f, 0, sizeof(*f));

   f->instance.vk.base.type = VK_OBJECT_TYPE_INSTANCE;
   f->instance.vk.base.client_visible = true;
   list_inithead(&f->instance.vk.debug_utils.callbacks);
   list_inithead(&f->instance.vk.debug_report.callbacks);

   f->pdev.vk.base.type = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
   f->pdev.vk.base.client_visible = true;
   f->pdev.vk.instance = &f->instance.vk;
   f->pdev.dev.chip = chip;

   f->dev.vk.base.type = VK_OBJECT_TYPE_DEVICE;
   f->dev.vk.base.device = &f->dev.vk;
   f->dev.vk.base.client_visible = true;
   f->dev.vk.physical = &f->pdev.vk;
   f->dev.vk.alloc = *vk_default_allocator();
   f->dev.dev.chip = chip;
   f->dev.dev.main_heap_base = UINT64_C(0x10000000);
   f->dev.dev.main_heap_size = UINT64_C(0x10000000);
   f->dev.dev.sparse_ro_offset = UINT64_C(1) << 39;

   simple_mtx_init(&f->dev.sparse_buffers.lock, mtx_plain);
   list_inithead(&f->dev.sparse_buffers.list);

   if (heap_entries) {
      f->heap = calloc(heap_entries, sizeof(*f->heap));
      assert(f->heap);
      simple_mtx_init(&f->dev.sampled_images.mutex, mtx_plain);
      f->dev.sampled_images.desc_size = sizeof(*f->heap);
      f->dev.sampled_images.alloc = heap_entries;
      f->dev.sampled_images.max_alloc = heap_entries;
      f->dev.sampled_images.next_desc = 1;
      f->dev.sampled_images.map = f->heap;
      f->dev.sampled_images.free_table =
         calloc(heap_entries, sizeof(uint32_t));
      f->dev.sampled_images.replay_free_table =
         calloc(heap_entries, sizeof(uint32_t));
      assert(f->dev.sampled_images.free_table &&
             f->dev.sampled_images.replay_free_table);
   }
}

static void
fixture_finish(struct test_fixture *f)
{
   hk_texel_windows_finish(&f->dev);
   if (f->heap) {
      free(f->dev.sampled_images.free_table);
      free(f->dev.sampled_images.replay_free_table);
      simple_mtx_destroy(&f->dev.sampled_images.mutex);
      free(f->heap);
   }
   simple_mtx_destroy(&f->dev.sparse_buffers.lock);
}

static void
init_buffer(struct hk_buffer *buf, struct hk_device *dev, uint64_t addr,
            uint64_t size)
{
   memset(buf, 0, sizeof(*buf));
   buf->vk.base.type = VK_OBJECT_TYPE_BUFFER;
   buf->vk.base.device = &dev->vk;
   buf->vk.base.client_visible = true;
   buf->vk.device_address = addr;
   buf->vk.size = size;
}

static void
check_agx2_chip(enum agx_chip chip)
{
   const unsigned gen = agx_chip_descriptor_generation(chip);
   assert(hk_descriptor_generation_is_agx2(gen));

   struct test_fixture f;
   fixture_init(&f, chip, 0);

   struct hk_native_texture_descriptor null_tex;
   struct hk_native_pbe_descriptor null_pbe;
   assert(hk_set_null_native_texture(gen, &null_tex));
   assert(hk_set_null_native_pbe(gen, &null_pbe));

   const struct hk_storage_image_descriptor_layout layout =
      hk_storage_image_descriptor_layout(gen);
   const unsigned tex_size = hk_texture_descriptor_size(gen);
   const unsigned pbe_size = hk_pbe_descriptor_size(gen);
   const uint64_t base_addr = UINT64_C(0x1000000000);

   struct hk_buffer buf;
   init_buffer(&buf, &f.dev, base_addr, UINT64_C(0x400000000));

   /* 1. AGX2 texel buffer height must include + tail_offset_el. */
   const struct {
      VkFormat format;
      uint32_t blocksize_B;
      uint64_t offset;
      uint64_t elements;
      uint32_t expected_height;
      uint32_t expected_tail_el;
   } height_cases[] = {
      {VK_FORMAT_R32_UINT, 4, 0, 16384, 1, 0},
      {VK_FORMAT_R32_UINT, 4, 4, 16384, 2, 1},
      {VK_FORMAT_R32_UINT, 4, 4, 16383, 1, 1},
      {VK_FORMAT_R32_UINT, 4, 12, 16384, 2, 3},
      {VK_FORMAT_R8_UINT, 1, 15, 16384, 2, 15},
      {VK_FORMAT_R8_UINT, 1, 15, AGX_TEXTURE_BUFFER_MAX_SIZE - 15,
       AGX_TEXTURE_BUFFER_MAX_HEIGHT, 15},
      {VK_FORMAT_R32_UINT, 4, 4, AGX_TEXTURE_BUFFER_MAX_SIZE - 1,
       AGX_TEXTURE_BUFFER_MAX_HEIGHT, 1},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(height_cases); ++i) {
      const VkBufferViewCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = hk_buffer_to_handle(&buf),
         .format = height_cases[i].format,
         .offset = height_cases[i].offset,
         .range = height_cases[i].elements * height_cases[i].blocksize_B,
      };
      VkBufferView bv_handle = VK_NULL_HANDLE;
      VkResult res = hk_CreateBufferView(hk_device_to_handle(&f.dev), &ci, NULL,
                                         &bv_handle);
      assert(res == VK_SUCCESS);

      VK_FROM_HANDLE(hk_buffer_view, view, bv_handle);
      struct AGX_TEXTURE tex_u;
      struct AGX_PBE pbe_u;
      AGX_TEXTURE_unpack(NULL, (const uint8_t *)view->tex.opaque, &tex_u);
      AGX_PBE_unpack(NULL, (const uint8_t *)view->pbe.opaque, &pbe_u);

      assert(tex_u.width == AGX_TEXTURE_BUFFER_WIDTH);
      assert(tex_u.height == height_cases[i].expected_height);
      assert(tex_u.address == base_addr);
      assert(tex_u.buffer_size_sw == height_cases[i].elements);
      assert(tex_u.buffer_offset_sw == height_cases[i].expected_tail_el);

      assert(pbe_u.width == AGX_TEXTURE_BUFFER_WIDTH);
      assert(pbe_u.height == height_cases[i].expected_height);
      assert(pbe_u.buffer == base_addr);
      assert(pbe_u.buffer_offset_sw == height_cases[i].expected_tail_el);

      /* Descriptor-buffer path must match byte for byte. */
      uint8_t packed[64] = {0};
      const VkDescriptorAddressInfoEXT info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
         .address = base_addr + height_cases[i].offset,
         .range = height_cases[i].elements * height_cases[i].blocksize_B,
         .format = height_cases[i].format,
      };
      hk_texel_buffer_descriptor(&f.dev, &info, true, packed);
      assert(memcmp(packed + layout.texture, &view->tex, tex_size) == 0);
      assert(memcmp(packed + layout.pbe, &view->pbe, pbe_size) == 0);

      memset(packed, 0, sizeof(packed));
      hk_texel_buffer_descriptor(&f.dev, &info, false, packed);
      assert(memcmp(packed + layout.texture, &view->tex, tex_size) == 0);
      assert(memcmp(packed + layout.pbe, &null_pbe, pbe_size) == 0);

      hk_DestroyBufferView(hk_device_to_handle(&f.dev), bv_handle, NULL);
   }

   /* 2. Max-height boundary rejection when elements + tail_offset_el crosses
    * AGX_TEXTURE_BUFFER_MAX_SIZE (would require height == 16385).
    */
   const struct {
      VkFormat format;
      uint32_t blocksize_B;
      uint64_t offset;
      uint64_t elements;
   } overflow_cases[] = {
      {VK_FORMAT_R32_UINT, 4, 4, AGX_TEXTURE_BUFFER_MAX_SIZE},
      {VK_FORMAT_R8_UINT, 1, 15, AGX_TEXTURE_BUFFER_MAX_SIZE - 14},
      {VK_FORMAT_R32_UINT, 4, 0, AGX_TEXTURE_BUFFER_MAX_SIZE + 1},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(overflow_cases); ++i) {
      const VkBufferViewCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = hk_buffer_to_handle(&buf),
         .format = overflow_cases[i].format,
         .offset = overflow_cases[i].offset,
         .range = overflow_cases[i].elements * overflow_cases[i].blocksize_B,
      };
      VkBufferView bv_handle = VK_NULL_HANDLE;
      assert(hk_CreateBufferView(hk_device_to_handle(&f.dev), &ci, NULL,
                                 &bv_handle) == VK_ERROR_FORMAT_NOT_SUPPORTED);
      assert(bv_handle == VK_NULL_HANDLE);

      uint8_t packed[64] = {0};
      const VkDescriptorAddressInfoEXT info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
         .address = base_addr + overflow_cases[i].offset,
         .range = overflow_cases[i].elements * overflow_cases[i].blocksize_B,
         .format = overflow_cases[i].format,
      };
      hk_texel_buffer_descriptor(&f.dev, &info, true, packed);
      assert(memcmp(packed + layout.texture, &null_tex, tex_size) == 0);
      assert(memcmp(packed + layout.pbe, &null_pbe, pbe_size) == 0);
   }

   /* Huge ranges must not overflow uint32_t/uint64_t height arithmetic. */
   {
      uint8_t packed[64] = {0};
      const VkDescriptorAddressInfoEXT info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
         .address = base_addr + 4,
         .range = UINT64_MAX & ~UINT64_C(0xf),
         .format = VK_FORMAT_R32_UINT,
      };
      hk_texel_buffer_descriptor(&f.dev, &info, true, packed);
      assert(memcmp(packed + layout.texture, &null_tex, tex_size) == 0);
      assert(memcmp(packed + layout.pbe, &null_pbe, pbe_size) == 0);
   }

   /* 3. AGX2 4-byte-aligned RGB32 uniform texel buffers (tail in {0,4,8,12}). */
   const VkFormat rgb32_formats[] = {
      VK_FORMAT_R32G32B32_SFLOAT,
      VK_FORMAT_R32G32B32_UINT,
      VK_FORMAT_R32G32B32_SINT,
   };
   const uint32_t rgb32_offsets[] = {0, 4, 8, 12};

   for (unsigned fi = 0; fi < ARRAY_SIZE(rgb32_formats); ++fi) {
      for (unsigned oi = 0; oi < ARRAY_SIZE(rgb32_offsets); ++oi) {
         const uint32_t tail = rgb32_offsets[oi];
         const uint64_t elements = 16384;
         const VkBufferViewCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .buffer = hk_buffer_to_handle(&buf),
            .format = rgb32_formats[fi],
            .offset = tail,
            .range = elements * 12,
         };
         VkBufferView bv_handle = VK_NULL_HANDLE;
         assert(hk_CreateBufferView(hk_device_to_handle(&f.dev), &ci, NULL,
                                    &bv_handle) == VK_SUCCESS);

         VK_FROM_HANDLE(hk_buffer_view, view, bv_handle);
         struct AGX_TEXTURE tex_u;
         AGX_TEXTURE_unpack(NULL, (const uint8_t *)view->tex.opaque, &tex_u);

         assert(tex_u.channels == AGX_CHANNELS_R32G32B32_EMULATED);
         assert(tex_u.address == base_addr);
         assert((tex_u.address & 0xf) == 0);
         assert(tex_u.buffer_size_sw == elements);
         assert(tex_u.buffer_offset_sw == tail / 4);
         assert(tex_u.width == AGX_TEXTURE_BUFFER_WIDTH);
         assert(tex_u.height == 1);
         assert(memcmp(&view->pbe, &null_pbe, pbe_size) == 0);

         /* Descriptor-buffer uniform path produces the same non-null texture
          * descriptor and null PBE.
          */
         uint8_t packed[64] = {0};
         const VkDescriptorAddressInfoEXT info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
            .address = base_addr + tail,
            .range = elements * 12,
            .format = rgb32_formats[fi],
         };
         hk_texel_buffer_descriptor(&f.dev, &info, false, packed);
         assert(memcmp(packed + layout.texture, &null_tex, tex_size) != 0);
         assert(memcmp(packed + layout.texture, &view->tex, tex_size) == 0);
         assert(memcmp(packed + layout.pbe, &null_pbe, pbe_size) == 0);

         /* RGB32 has no storage support: PBE stays null even if requested. */
         memset(packed, 0, sizeof(packed));
         hk_texel_buffer_descriptor(&f.dev, &info, true, packed);
         assert(memcmp(packed + layout.texture, &view->tex, tex_size) == 0);
         assert(memcmp(packed + layout.pbe, &null_pbe, pbe_size) == 0);

         hk_DestroyBufferView(hk_device_to_handle(&f.dev), bv_handle, NULL);
      }
   }

   /* Zero-element RGB32 view (range < 12) stays valid with height 1 and size 0. */
   {
      uint8_t packed[64] = {0};
      const VkDescriptorAddressInfoEXT info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
         .address = base_addr + 4,
         .range = 4,
         .format = VK_FORMAT_R32G32B32_UINT,
      };
      hk_texel_buffer_descriptor(&f.dev, &info, false, packed);
      struct AGX_TEXTURE tex_u;
      AGX_TEXTURE_unpack(NULL, packed + layout.texture, &tex_u);
      assert(tex_u.channels == AGX_CHANNELS_R32G32B32_EMULATED);
      assert(tex_u.address == base_addr);
      assert(tex_u.buffer_size_sw == 0);
      assert(tex_u.buffer_offset_sw == 1);
      assert(tex_u.height == 1);
   }

   /* 4. Unaligned offsets must be rejected by hk_texel_buffer_descriptor(). */
   for (uint32_t tail = 1; tail < 16; ++tail) {
      if (tail % 4 == 0)
         continue;
      uint8_t packed[64] = {0};
      const VkDescriptorAddressInfoEXT info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
         .address = base_addr + tail,
         .range = 12 * 16,
         .format = VK_FORMAT_R32G32B32_SFLOAT,
      };
      hk_texel_buffer_descriptor(&f.dev, &info, false, packed);
      assert(memcmp(packed + layout.texture, &null_tex, tex_size) == 0);
      assert(memcmp(packed + layout.pbe, &null_pbe, pbe_size) == 0);
   }

   /* Non-RGB32 still requires full texel alignment (e.g. 8B for R32G32). */
   {
      uint8_t packed[64] = {0};
      const VkDescriptorAddressInfoEXT info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
         .address = base_addr + 4,
         .range = 8 * 16,
         .format = VK_FORMAT_R32G32_UINT,
      };
      hk_texel_buffer_descriptor(&f.dev, &info, false, packed);
      assert(memcmp(packed + layout.texture, &null_tex, tex_size) == 0);
      assert(memcmp(packed + layout.pbe, &null_pbe, pbe_size) == 0);
   }

   /* 5. Sparse buffer address applies sparse_ro_offset to sampled address
    * while preserving the 4-byte RGB32 word tail and 16-byte address alignment.
    */
   {
      struct agx_va sparse_va = {.addr = base_addr};
      struct hk_buffer sparse_buf;
      init_buffer(&sparse_buf, &f.dev, base_addr, UINT64_C(0x100000));
      sparse_buf.va = &sparse_va;
      list_addtail(&sparse_buf.sparse_link, &f.dev.sparse_buffers.list);
      p_atomic_set(&f.dev.sparse_buffers.count, 1);

      uint8_t packed[64] = {0};
      const VkDescriptorAddressInfoEXT info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
         .address = base_addr + 8,
         .range = 12 * 32,
         .format = VK_FORMAT_R32G32B32_UINT,
      };
      hk_texel_buffer_descriptor(&f.dev, &info, false, packed);
      struct AGX_TEXTURE tex_u;
      AGX_TEXTURE_unpack(NULL, packed + layout.texture, &tex_u);
      assert(tex_u.channels == AGX_CHANNELS_R32G32B32_EMULATED);
      assert(tex_u.address == base_addr + f.dev.dev.sparse_ro_offset);
      assert(tex_u.buffer_size_sw == 32);
      assert(tex_u.buffer_offset_sw == 2);

      list_del(&sparse_buf.sparse_link);
      p_atomic_set(&f.dev.sparse_buffers.count, 0);
   }

   fixture_finish(&f);
}

static void
check_agx3_g17p_unchanged(void)
{
   struct test_fixture f;
   fixture_init(&f, AGX_CHIP_G17P, 1024);
   assert(hk_texel_windows_init(&f.dev) == VK_SUCCESS);

   const unsigned gen = agx_chip_descriptor_generation(AGX_CHIP_G17P);
   assert(gen == 9);

   struct hk_native_pbe_descriptor null_pbe;
   assert(hk_set_null_native_pbe(gen, &null_pbe));

   const uint64_t base_addr = f.dev.dev.main_heap_base + 0x1000;
   struct hk_buffer buf;
   init_buffer(&buf, &f.dev, base_addr, UINT64_C(0x100000));

   /* G17P hk_CreateBufferView: R32_UINT with tail_offset_el = 1, elements = 16384 */
   {
      const VkBufferViewCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = hk_buffer_to_handle(&buf),
         .format = VK_FORMAT_R32_UINT,
         .offset = 4,
         .range = 16384 * 4,
      };
      VkBufferView bv_handle = VK_NULL_HANDLE;
      assert(hk_CreateBufferView(hk_device_to_handle(&f.dev), &ci, NULL,
                                 &bv_handle) == VK_SUCCESS);
      VK_FROM_HANDLE(hk_buffer_view, view, bv_handle);

      assert(view->buffer_metadata.elements == 16384);
      assert(view->buffer_metadata.offset_el == 1);
      assert(view->buffer_metadata.address == base_addr + 4);
      assert(view->buffer_metadata.rgb32 == 0);
      assert(view->sampled_heap_index > 0);

      struct AGX_TEXTURE_GEN9 tex9;
      assert(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)view->tex.opaque,
                                     &tex9));
      assert(tex9.address == base_addr);
      assert(tex9.width == AGX_TEXTURE_BUFFER_WIDTH);
      assert(tex9.height == 2);

      struct agx_pbe_agx3_2d_fields pbe9;
      assert(agx_pbe_agx3_2d_unpack(gen, view->pbe.opaque,
                                    sizeof(view->pbe.opaque), &pbe9));
      assert(pbe9.buffer == base_addr);
      assert(pbe9.width == AGX_TEXTURE_BUFFER_WIDTH);
      assert(pbe9.height == 2);

      hk_DestroyBufferView(hk_device_to_handle(&f.dev), bv_handle, NULL);
   }

   /* G17P hk_CreateBufferView: 4-byte-aligned RGB32 keeps metadata address */
   {
      const VkBufferViewCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = hk_buffer_to_handle(&buf),
         .format = VK_FORMAT_R32G32B32_SFLOAT,
         .offset = 4,
         .range = 16384 * 12,
      };
      VkBufferView bv_handle = VK_NULL_HANDLE;
      assert(hk_CreateBufferView(hk_device_to_handle(&f.dev), &ci, NULL,
                                 &bv_handle) == VK_SUCCESS);
      VK_FROM_HANDLE(hk_buffer_view, view, bv_handle);

      assert(view->buffer_metadata.elements == 16384);
      assert(view->buffer_metadata.offset_el == 0);
      assert(view->buffer_metadata.address == base_addr + 4);
      assert(view->buffer_metadata.rgb32 == 1);

      struct AGX_TEXTURE_GEN9 tex9;
      assert(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)view->tex.opaque,
                                     &tex9));
      assert(tex9.channels == AGX_CHANNELS_R32);
      assert(tex9.address == base_addr);
      assert(tex9.height == 1);
      assert(memcmp(&view->pbe, &null_pbe, sizeof(null_pbe)) == 0);

      hk_DestroyBufferView(hk_device_to_handle(&f.dev), bv_handle, NULL);
   }

   /* G17P hk_texel_buffer_descriptor: RGB32 at +8 keeps exact metadata address */
   {
      uint8_t packed[AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE] = {0};
      const VkDescriptorAddressInfoEXT info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
         .address = base_addr + 8,
         .range = 64 * 12,
         .format = VK_FORMAT_R32G32B32_UINT,
      };
      hk_texel_buffer_descriptor(&f.dev, &info, false, packed);

      struct hk_g17p_buffer_texture_metadata meta;
      memcpy(&meta, packed + AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET,
             sizeof(meta));
      assert(meta.elements == 64);
      assert(meta.offset_el == 0);
      assert(meta.address == base_addr + 8);
      assert(meta.rgb32 == 1);
   }

   fixture_finish(&f);
}

int
main(void)
{
   check_agx2_chip(AGX_CHIP_G13G);
   check_agx2_chip(AGX_CHIP_G14G);
   check_agx3_g17p_unchanged();
   puts("honeykrisp buffer view descriptors: passed");
   return 0;
}
