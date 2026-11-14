/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The G17P kernel-end SCS sweep and its switch, ASAHI_G17P_RQ_SCS_SWEEP.
 *
 * Lowering: which destination-less loads a candidate kernel ends with for
 * each mask, in which order, between which fences, and that general NIR
 * cleanup keeps them although nothing reads them. Without the variable there
 * are none. Only a thread that launched a query issues them, unless
 * ASAHI_G17P_RQ_SCS_SWEEP_SKIP=0 gives them to every thread.
 *
 * Launch state: the query context and the constant words of the warp's save
 * area are stored under a branch on the thread's first launch
 * (ASAHI_G17P_RQ_CTX_ONCE). Where a kernel's only launch site is outside any
 * loop the branch folds away, and the kernel is the same with and without it.
 *
 * Back end: a kernel carrying such loads compiles for G17P, its binary holds
 * exactly those accesses in the destination-less form between the two
 * traversal barriers, and no ordinary SCS load takes their place. The lowered
 * query kernel itself compiles with every mask and ends with its sweep.
 */
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx_nir_gen1_ray_query.h"
#include "agx3_rt_core.h"
#include "agx3_scs_core.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

#define SWEEP "ASAHI_G17P_RQ_SCS_SWEEP"
#define EXTRA "ASAHI_G17P_RQ_SCS_SWEEP_EXTRA"
#define SKIP "ASAHI_G17P_RQ_SCS_SWEEP_SKIP"
#define ONCE "ASAHI_G17P_RQ_CTX_ONCE"
#define RESULT_ADDRESS 0x10004000000ull

struct touch {
   bool indexed;
   int offset;
   unsigned words, shift;
};

/* What a warp's `bytes` of storage take: whole 512-byte rows of four words a
 * lane, then a remainder of one word a lane. */
static unsigned
rows(struct touch *out, unsigned n, unsigned bytes, bool indexed)
{
   for (unsigned at = 0; at + 512 <= bytes; at += 512)
      out[n++] = (struct touch){indexed, at, 4, indexed ? 4 : 0};
   if (bytes % 512)
      out[n++] = (struct touch){indexed, bytes - bytes % 512, 1,
                                indexed ? 2 : 0};
   return n;
}

/* The documented sweep for a mask and extra per-thread sizes. */
static unsigned
expected(unsigned mask, const unsigned *extra, unsigned extras,
         struct touch *out)
{
   const bool indexed = !(mask & 16);
   unsigned n = 0;
   if (mask & 2)
      n = rows(out, n, 32 * 32, indexed);
   for (unsigned i = 0; (mask & 32) && i < extras; ++i)
      n = rows(out, n, 32 * extra[i], indexed);
   if (mask & 64) {
      n = rows(out, n, 32 * 32, indexed);
      n = rows(out, n, 32 * 16, indexed);
   }
   if (mask & 4) {
      out[n++] = (struct touch){false, 256, 4, 0};
      out[n++] = (struct touch){false, 272, 4, 0};
      out[n++] = (struct touch){false, 288, 2, 0};
   }
   if (mask & 8) {
      for (unsigned i = 0; i < 6; ++i)
         out[n++] = (struct touch){false, 16 * i, 4, 0};
      out[n++] = (struct touch){false, 104, 2, 0};
   }
   if (mask & 1)
      n = rows(out, n, 9152 + 32 * 6, indexed);
   return n;
}

static bool
is_touch(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return (intr->intrinsic == nir_intrinsic_load_scs_indexed_agx ||
           intr->intrinsic == nir_intrinsic_load_scs_base_agx) &&
          (nir_intrinsic_flags(intr) & AGX_NIR_SCS_TOUCH);
}

static void
set_arm(const char *mask, const char *extra)
{
   if (mask)
      setenv(SWEEP, mask, 1);
   else
      unsetenv(SWEEP);
   if (extra)
      setenv(EXTRA, extra, 1);
   else
      unsetenv(EXTRA);
}

/* A one-query candidate kernel, lowered under the current environment. Its
 * result store sits inside control flow, as a real kernel's does. */
static nir_builder
lowered_kernel(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options, "Gen1 SCS sweep");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.ray_queries = 1;
   nir_variable *query = nir_local_variable_create(b.impl,
      glsl_uint64_t_type(), "query");
   query->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(&b, query)->def;
   nir_def *id = nir_load_local_invocation_index(&b);
   nir_rq_initialize(&b, rq, nir_imm_int64(&b, 0x10003000000ull),
      nir_imm_int(&b, 1), nir_imm_int(&b, 255), nir_imm_vec3(&b, 0, 0, 1),
      nir_imm_float(&b, 0.0001), nir_imm_vec3(&b, 0, 0, -1),
      nir_imm_float(&b, 9));
   nir_def *more = nir_rq_proceed(&b, 1, rq);
   nir_push_if(&b, nir_inot(&b, more));
   nir_store_global(&b, nir_iadd_imm(&b, id, 7),
      nir_imm_int64(&b, RESULT_ADDRESS), .align_mul = 4, .write_mask = 1);
   nir_pop_if(&b, NULL);
   nir_shader_gather_info(b.shader, b.impl);
   const struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = 118, .bvh_uniform = 96, .status_uniform = 128,
      .context_uniform = 160, .as_table_uniform = 164, .as_table_first = 1,
      .launch_header = 0x210000, .instance_traversal = true,
      .candidates = true, .compute_kernel = true,
   };
   CHECK(agx_nir_lower_gen1_ray_query(b.shader, &layout));
   return b;
}

/* Lower the kernel under the given environment and compare its sweep with
 * `want`. */
static void
check_lowering(const char *mask, const char *extra, const struct touch *want,
               unsigned count)
{
   set_arm(mask, extra);
   nir_builder b = lowered_kernel();
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(progress, b.shader, nir_opt_copy_prop);
      NIR_PASS(progress, b.shader, nir_opt_constant_folding);
      NIR_PASS(progress, b.shader, nir_opt_dead_cf);
      NIR_PASS(progress, b.shader, nir_opt_cse);
      NIR_PASS(progress, b.shader, nir_opt_dce);
   } while (progress);
   nir_validate_shader(b.shader, "kernel-end sweep");

   /* Every touch is in one block, after everything else that block does, and
    * nothing follows the first one but touches. With every thread sweeping
    * it is the block all threads end in. Otherwise that block is empty and
    * the touches are all there is under the branch ahead of it, which is
    * taken by the threads that launched. */
   nir_block *last = nir_impl_last_block(b.impl);
   const char *every = getenv(SKIP);
   if (count && !(every && !strcmp(every, "0"))) {
      nir_cf_node *prev = nir_cf_node_prev(&last->cf_node);
      CHECK(exec_list_is_empty(&last->instr_list));
      CHECK(prev && prev->type == nir_cf_node_if);
      nir_if *launched = nir_cf_node_as_if(prev);
      CHECK(!nir_src_is_const(launched->condition));
      CHECK(nir_if_first_else_block(launched) ==
            nir_if_last_else_block(launched));
      CHECK(exec_list_is_empty(
         &nir_if_first_else_block(launched)->instr_list));
      last = nir_if_first_then_block(launched);
      CHECK(last == nir_if_last_then_block(launched));
   }
   unsigned seen = 0, elsewhere = 0, stores = 0, launches = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (block != last)
            elsewhere += is_touch(instr);
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         stores += intr->intrinsic == nir_intrinsic_store_global &&
                   nir_src_is_const(intr->src[1]) &&
                   nir_src_as_uint(intr->src[1]) == RESULT_ADDRESS;
         launches += intr->intrinsic == nir_intrinsic_intersect_ray_gen1_agx;
      }
   }
   CHECK(elsewhere == 0 && stores == 1 && launches == 1);

   nir_foreach_instr(instr, last) {
      if (!is_touch(instr)) {
         CHECK(seen == 0);
         CHECK(instr->type != nir_instr_type_intrinsic ||
               nir_instr_as_intrinsic(instr)->intrinsic !=
                  nir_intrinsic_store_global);
         continue;
      }
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      CHECK(seen < count);
      const struct touch *t = &want[seen];
      const bool indexed =
         intr->intrinsic == nir_intrinsic_load_scs_indexed_agx;
      const unsigned fences =
         (seen == 0 ? AGX_NIR_SCS_TOUCH_FENCE_BEFORE : 0) |
         (seen + 1 == count ? AGX_NIR_SCS_TOUCH_FENCE_AFTER : 0);
      CHECK(indexed == t->indexed);
      CHECK(nir_intrinsic_base(intr) == t->offset);
      CHECK(intr->def.num_components == t->words && intr->def.bit_size == 32);
      CHECK(nir_intrinsic_component(intr) == t->shift);
      CHECK(nir_intrinsic_flags(intr) == (AGX_NIR_SCS_TOUCH | fences));
      CHECK(nir_def_is_unused(&intr->def));
      CHECK(intr->src[0].ssa->bit_size == 32);
      if (indexed)
         CHECK(intr->src[1].ssa->bit_size == 32);
      else
         CHECK(nir_intrinsic_format(intr) == PIPE_FORMAT_R32_UINT);
      ++seen;
   }
   CHECK(seen == count);
   ralloc_free(b.shader);
}

static void
check_mask(unsigned mask, const char *extra, const unsigned *sizes,
           unsigned extras)
{
   struct touch want[64];
   char text[16];
   snprintf(text, sizeof(text), "0x%x", mask);
   const unsigned count = expected(mask, sizes, extras, want);
   check_lowering(text, extra, want, count);
}

struct scan {
   unsigned loads;       /* ordinary SCS loads */
   unsigned fences[2];   /* traversal barriers: launch scope, end scope */
   unsigned fence_at[2]; /* offset of the last of each */
   unsigned first_at, last_at;
};

/* Find every destination-less load of a compiled kernel and compare them, in
 * order, with `want`. */
static struct scan
scan_binary(const struct agx_shader_part *part, enum agx_isa_encoding enc,
            const struct touch *want, unsigned count)
{
   uint8_t fence[2][AGX3_RT_CORE_MAX_LENGTH], fence_length[2];
   CHECK(agx3_rt_traversal_barrier(enc, AGX3_RT_BARRIER_ENTRY, fence[0],
                                   &fence_length[0]));
   CHECK(agx3_rt_traversal_barrier(enc, AGX3_RT_BARRIER_IN_LOOP, fence[1],
                                   &fence_length[1]));
   const uint8_t *code = (const uint8_t *)part->binary;
   const unsigned begin = part->info.main_offset;
   const unsigned end = begin + part->info.main_size;
   struct scan found = {0};
   unsigned seen = 0;
   for (unsigned i = begin; i + 2 <= end; i += 2) {
      for (unsigned f = 0; f < 2; ++f) {
         if (i + fence_length[f] <= end &&
             !memcmp(code + i, fence[f], fence_length[f])) {
            found.fences[f]++;
            found.fence_at[f] = i;
         }
      }
      if (i + 14 > end)
         continue;
      struct agx3_scs_touch_desc d;
      struct agx3_scs_desc load;
      found.loads += agx3_scs_decode_load(enc, code + i, 14, &load);
      if (!agx3_scs_decode_touch(enc, code + i, 14, &d))
         continue;
      CHECK(seen < count);
      const struct touch *t = &want[seen];
      CHECK(d.has_index == t->indexed);
      CHECK(d.byte_offset == t->offset);
      CHECK(d.component_mask == (1u << t->words) - 1);
      CHECK(d.address_selector == t->shift);
      CHECK(!d.index_kill && !d.base_kill);
      CHECK(!(d.base_register & 1) && (!d.has_index || !(d.index & 1)));
      CHECK(!d.has_index || d.index != d.base_register);
      if (seen == 0)
         found.first_at = i;
      found.last_at = i;
      ++seen;
   }
   CHECK(seen == count);
   return found;
}

/* Compile a kernel that ends with the given touches and check its binary. */
enum fault {
   FAULT_NONE,
   FAULT_USED,      /* the access has no value to read */
   FAULT_SELECTOR,  /* only four words or one word a lane */
   FAULT_OFFSET,    /* word-aligned offsets only */
   FAULT_WORDS,     /* one, two or four words */
   FAULT_FLAGS,     /* no other load flag combines with a touch */
   FAULT_ENCODING,  /* G17P only */
};

static bool
compile_touches(const struct touch *want, unsigned count, enum fault fault)
{
   const bool g17p = fault != FAULT_ENCODING;
   const enum agx_isa_encoding enc = g17p ? AGX_ISA_ENCODING_AGX3B
                                          : AGX_ISA_ENCODING_AGX3C;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options, "Gen1 SCS touch");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *id = nir_load_local_invocation_index(&b);
   nir_def *lane = nir_iand_imm(&b, id, 31);
   nir_def *warp = nir_ushr_imm(&b, id, 5);
   nir_def *block = nir_iadd_imm(&b, nir_imul_imm(&b, warp, 304 * 32),
                                 0x2f8000);
   nir_def *record = nir_iadd_imm(&b, nir_imul_imm(&b, warp, 48), 0x6000);
   nir_def *address = nir_iadd_imm(&b,
      nir_u2u64(&b, nir_imul_imm(&b, id, 16)), 0x20d0020000ull);
   nir_store_global(&b, nir_iadd_imm(&b, id, 23), address, .align_mul = 16,
                    .write_mask = 1);
   for (unsigned i = 0; i < count; ++i) {
      const struct touch *t = &want[i];
      const bool first = i == 0, final = i + 1 == count;
      unsigned flags = AGX_NIR_SCS_TOUCH |
         (first ? AGX_NIR_SCS_TOUCH_FENCE_BEFORE : 0) |
         (final ? AGX_NIR_SCS_TOUCH_FENCE_AFTER : 0);
      unsigned words = t->words, shift = t->shift;
      int offset = t->offset;
      if (final && fault == FAULT_SELECTOR)
         shift = 3;
      if (final && fault == FAULT_OFFSET)
         offset += 2;
      if (final && fault == FAULT_WORDS)
         words = 3;
      if (final && fault == FAULT_FLAGS)
         flags |= 1;
      nir_def *value = t->indexed
         ? nir_load_scs_indexed_agx(&b, words, 32, block, lane,
                                    .base = offset, .component = shift,
                                    .flags = flags)
         : nir_load_scs_base_agx(&b, words, 32, record, .base = offset,
                                 .format = PIPE_FORMAT_R32_UINT,
                                 .component = shift, .flags = flags);
      if (final && fault == FAULT_USED)
         nir_store_global(&b, nir_channel(&b, value, 0),
                          nir_iadd_imm(&b, address, 4), .align_mul = 4,
                          .write_mask = 1);
   }
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev = {.chip = g17p ? AGX_CHIP_G17P : AGX_CHIP_G17S, .encoding = enc},
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   printf("touches=%u fault=%u: %s %s\n", count, fault,
          ok ? "compiled" : "refused", ok ? "" : agx_last_compile_refusal());
   if (ok) {
      /* No ordinary SCS load, and one fence of each scope: the launch scope
       * ahead of the first touch, the other behind the last. */
      const struct scan found = scan_binary(&part, enc, want, count);
      CHECK(found.loads == 0);
      CHECK(found.fences[0] == 1 && found.fences[1] == 1);
      CHECK(found.fence_at[0] < found.first_at &&
            found.fence_at[1] > found.last_at);
      CHECK(part.info.uses_ray_traversal);
      printf("bytes=%u gprs=%u\n", part.info.main_size, part.info.nr_gprs);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/* The lowered query kernel through the back end. Returns its size. */
static unsigned
compile_kernel(unsigned mask, const char *extra, const unsigned *sizes,
               unsigned extras, struct scan *found)
{
   struct touch want[64];
   char text[16];
   snprintf(text, sizeof(text), "0x%x", mask);
   const unsigned count = expected(mask, sizes, extras, want);
   set_arm(text, extra);
   nir_builder b = lowered_kernel();
   set_arm(NULL, NULL);

   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev = {.chip = AGX_CHIP_G17P, .encoding = AGX_ISA_ENCODING_AGX3B},
      .no_preamble = true, .has_scratch = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   printf("kernel mask=0x%x extras=%u: %s %s\n", mask, extras,
          ok ? "compiled" : "refused", ok ? "" : agx_last_compile_refusal());
   CHECK(ok && part.info.uses_ray_traversal && part.info.main_size);
   *found = scan_binary(&part, AGX_ISA_ENCODING_AGX3B, want, count);
   const unsigned size = part.info.main_size;
   printf("bytes=%u gprs=%u touches=%u fences=%u/%u\n", size,
          part.info.nr_gprs, count, found->fences[0], found->fences[1]);
   free(part.binary);
   ralloc_free(b.shader);
   return size;
}

static unsigned
lowering_cases(void)
{
   unsigned cases = 0;

   /* Off: nothing, whatever the extra list says. */
   check_lowering("0", NULL, NULL, 0);
   check_lowering("0", "0x40000/16", NULL, 0);
   cases += 2;

   /* Unset: the payload records, the two other per-thread areas and the
    * query block, and no extra storage even when a list is given. The same
    * loads when every thread is to issue them. */
   {
      struct touch want[64];
      const unsigned count = expected(0x43, NULL, 0, want);
      CHECK(count == 2 + 2 + 1 + 19);
      check_lowering(NULL, NULL, want, count);
      check_lowering(NULL, "0x40000/16", want, count);
      setenv(SKIP, "0", 1);
      check_lowering(NULL, NULL, want, count);
      check_mask(0x7f, NULL, NULL, 0);
      check_lowering("0", NULL, NULL, 0);
      setenv(SKIP, "1", 1);
      check_lowering(NULL, NULL, want, count);
      unsetenv(SKIP);
      cases += 6;
   }

   /* Each store class alone, all of them, and both addressing forms. The
    * addressing bit selects nothing, nor does the extra bit without a list. */
   static const unsigned masks[] = {1, 2, 4, 8, 3, 0xf, 0x10, 0x11, 0x12,
                                    0x13, 0x1c, 0x1f, 0x20, 0x30, 0x3f,
                                    0x40, 0x43, 0x50, 0x53, 0x7f};
   for (unsigned i = 0; i < ARRAY_SIZE(masks); ++i, ++cases)
      check_mask(masks[i], NULL, NULL, 0);

   struct touch want[64];
   CHECK(expected(1, NULL, 0, want) == 19);
   CHECK(expected(2, NULL, 0, want) == 2);
   CHECK(expected(4, NULL, 0, want) == 3);
   CHECK(expected(8, NULL, 0, want) == 7);
   CHECK(expected(0xf, NULL, 0, want) == 31);
   CHECK(expected(0x30, NULL, 0, want) == 0);

   /* Extra per-thread storage follows the payload records, and only with
    * its bit. */
   static const unsigned two[] = {48, 16}, one[] = {16}, odd[] = {20, 4};
   check_mask(0x23, "0x10000/48,0x40000/16", two, 2);
   check_mask(0x33, "0x10000/48,0x40000/16", two, 2);
   check_mask(0x3f, "0x10000/48,0x40000/16", two, 2);
   check_mask(0x7f, "0x10000/48,0x40000/16", two, 2);
   check_mask(0x63, "0x10000/48,0x40000/16", two, 2);
   check_mask(0x20, "0x40000/16", one, 1);
   check_mask(0x30, "0x40000/16", one, 1);
   check_mask(0x21, "0x40000/20,0x50000/4", odd, 2);
   check_mask(3, "0x10000/48,0x40000/16", two, 2);
   check_mask(0x1f, "0x10000/48,0x40000/16", two, 2);
   cases += 10;
   CHECK(expected(0x40, NULL, 0, want) == 3);
   CHECK(expected(0x43, NULL, 0, want) == 24);
   CHECK(expected(0x23, two, 2, want) == 2 + 3 + 1 + 19);
   CHECK(expected(3, two, 2, want) == 21);

   /* A malformed list, or storage that does not end below the query
    * blocks, is dropped whole. */
   static const char *const malformed[] = {
      "0x10000", "0x10000/", "/32", "0x10000/33", "0x10001/32", "0x10000/0",
      "0x10000/32,", "0x10000/32;0x40000/16", "0x10000/512",
      "0x100000000/32", "0x2f8000/4", "0x2e0004/32", "0x40000/16,0x2f4000/16",
      "0/256,0x10000/256,0x30000/16",
   };
   for (unsigned i = 0; i < ARRAY_SIZE(malformed); ++i, ++cases)
      check_mask(0x23, malformed[i], NULL, 0);
   /* The last storage that fits: 3072 threads end at the query blocks. */
   static const unsigned fits[] = {16};
   check_mask(0x20, "0x2ec000/16", fits, 1);
   ++cases;
   unsetenv(SWEEP);
   unsetenv(EXTRA);
   return cases;
}

static unsigned
backend_cases(void)
{
   struct touch want[64];
   unsigned cases = 0;
   static const unsigned compiled[] = {1, 2, 4, 8, 0xf, 0x13, 0x1f};
   for (unsigned i = 0; i < ARRAY_SIZE(compiled); ++i, ++cases) {
      const unsigned count = expected(compiled[i], NULL, 0, want);
      CHECK(compile_touches(want, count, FAULT_NONE));
   }

   static const unsigned two[] = {48, 16};
   const unsigned count = expected(0x23, two, 2, want);
   CHECK(count == 25 && compile_touches(want, count, FAULT_NONE));
   ++cases;
   for (enum fault f = FAULT_USED; f <= FAULT_ENCODING; ++f, ++cases)
      CHECK(!compile_touches(want, count, f));

   /* The query kernel. Unset, it has no such load; set, it keeps every
    * traversal barrier it had and ends with one more of each scope around
    * its sweep, the last of them behind the last load. */
   struct scan off, on;
   const unsigned plain = compile_kernel(0, NULL, NULL, 0, &off);
   CHECK(off.fences[0] >= 1 && off.fences[1] >= 1);
   ++cases;
   static const unsigned kernel[] = {1, 2, 4, 8, 3, 0xf, 0x13, 0x1f, 0x40,
                                     0x43, 0x7f};
   for (unsigned i = 0; i < ARRAY_SIZE(kernel); ++i, ++cases) {
      const unsigned size = compile_kernel(kernel[i], NULL, NULL, 0, &on);
      CHECK(size > plain);
      CHECK(on.loads == off.loads);
      CHECK(on.fences[0] == off.fences[0] + 1);
      CHECK(on.fences[1] == off.fences[1] + 1);
      CHECK(on.fence_at[1] > on.last_at);
   }
   static const unsigned extra_masks[] = {0x23, 0x33, 0x20};
   for (unsigned i = 0; i < ARRAY_SIZE(extra_masks); ++i, ++cases) {
      compile_kernel(extra_masks[i], "0x10000/48,0x40000/16", two, 2, &on);
      CHECK(on.loads == off.loads);
      CHECK(on.fences[0] == off.fences[0] + 1);
      CHECK(on.fences[1] == off.fences[1] + 1);
      CHECK(on.fence_at[1] > on.last_at);
   }
   /* Bits that select no storage leave the kernel as it was. */
   static const unsigned idle[] = {0x10, 0x20, 0x30};
   for (unsigned i = 0; i < ARRAY_SIZE(idle); ++i, ++cases) {
      CHECK(compile_kernel(idle[i], NULL, NULL, 0, &on) == plain);
      CHECK(on.fences[0] == off.fences[0] && on.fences[1] == off.fences[1]);
   }

   /* With the variable unset the kernel is the one mask 0x43 gives. */
   {
      struct touch want[64];
      const unsigned count = expected(0x43, NULL, 0, want);
      const unsigned selected = compile_kernel(0x43, NULL, NULL, 0, &on);
      nir_builder b = lowered_kernel();
      agx_preprocess_nir(b.shader);
      struct agx_shader_key key = {
         .dev = {.chip = AGX_CHIP_G17P, .encoding = AGX_ISA_ENCODING_AGX3B},
         .no_preamble = true, .has_scratch = true,
      };
      struct agx_shader_part part = {0};
      CHECK(agx_compile_shader_nir(b.shader, &key, &part));
      const struct scan unset =
         scan_binary(&part, AGX_ISA_ENCODING_AGX3B, want, count);
      CHECK(part.info.main_size == selected && selected > plain);
      CHECK(unset.fences[0] == on.fences[0] && unset.fences[1] == on.fences[1]);
      CHECK(unset.loads == off.loads);
      free(part.binary);
      ralloc_free(b.shader);
      ++cases;
   }
   return cases;
}

/* Two queries one after the other in a thread, the second inside control
 * flow, lowered under the current environment. */
static nir_builder
two_query_kernel(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options, "Gen1 SCS sweep, two queries");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.ray_queries = 2;
   nir_def *id = nir_load_local_invocation_index(&b);
   nir_def *hit[2];
   for (unsigned q = 0; q < 2; ++q) {
      nir_variable *query = nir_local_variable_create(b.impl,
         glsl_uint64_t_type(), q ? "query1" : "query0");
      query->data.ray_query = true;
      nir_def *rq = &nir_build_deref_var(&b, query)->def;
      if (q)
         nir_push_if(&b, hit[0]);
      nir_rq_initialize(&b, rq, nir_imm_int64(&b, 0x10003000000ull),
         nir_imm_int(&b, q ? 4 : 1), nir_imm_int(&b, 255),
         nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 0.0001),
         nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 9));
      nir_def *more = nir_rq_proceed(&b, 1, rq);
      hit[q] = nir_inot(&b, more);
      if (q) {
         nir_store_global(&b, nir_iadd_imm(&b, id, 7),
            nir_imm_int64(&b, RESULT_ADDRESS), .align_mul = 4,
            .write_mask = 1);
         nir_pop_if(&b, NULL);
      }
   }
   nir_shader_gather_info(b.shader, b.impl);
   const struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = 118, .bvh_uniform = 96, .status_uniform = 128,
      .context_uniform = 160, .as_table_uniform = 164, .as_table_first = 1,
      .launch_header = 0x210000, .instance_traversal = true,
      .candidates = true, .compute_kernel = true,
   };
   CHECK(agx_nir_lower_gen1_ray_query(b.shader, &layout));
   return b;
}

struct thread_kernel {
   unsigned size;        /* compiled bytes */
   unsigned in_last;     /* touches in the block all threads end in */
   unsigned elsewhere;   /* touches anywhere else */
   unsigned branches;    /* if nodes of the lowered kernel */
   unsigned context;     /* stores of the query context */
};

static bool
is_context_store(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return intr->intrinsic == nir_intrinsic_store_scs_agx &&
          nir_intrinsic_base(intr) >= 0x6000 &&
          nir_intrinsic_base(intr) <= 0x6068;
}

/* The shape of a lowered kernel's launch state. */
static void
launch_shape(nir_function_impl *impl, struct thread_kernel *k)
{
   nir_foreach_block(block, impl) {
      nir_cf_node *prev = nir_cf_node_prev(&block->cf_node);
      k->branches += prev && prev->type == nir_cf_node_if;
      unsigned stores = 0;
      nir_foreach_instr(instr, block)
         stores += is_context_store(instr);
      /* A copy is seven stores in one block. */
      CHECK(stores == 0 || stores == 7);
      k->context += stores;
   }
}

/* Lower and compile the two-query kernel under the current environment. A
 * thread sweeps once, however many queries it launched. */
static struct thread_kernel
thread_kernel(const char *what)
{
   struct touch want[64];
   const unsigned count = expected(0x43, NULL, 0, want);
   nir_builder b = two_query_kernel();
   struct thread_kernel k = {0};
   unsigned launches = 0;
   nir_block *end = nir_impl_last_block(b.impl);
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (is_touch(instr))
            ++*(block == end ? &k.in_last : &k.elsewhere);
         launches += instr->type == nir_instr_type_intrinsic &&
            nir_instr_as_intrinsic(instr)->intrinsic ==
               nir_intrinsic_intersect_ray_gen1_agx;
      }
   }
   CHECK(launches == 2);
   CHECK(k.in_last + k.elsewhere == count);
   launch_shape(b.impl, &k);
   CHECK(k.context == 2 * 7);

   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev = {.chip = AGX_CHIP_G17P, .encoding = AGX_ISA_ENCODING_AGX3B},
      .no_preamble = true, .has_scratch = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   printf("two queries, %s: %s %s\n", what, ok ? "compiled" : "refused",
          ok ? "" : agx_last_compile_refusal());
   CHECK(ok && part.info.uses_ray_traversal && part.info.main_size);
   const struct scan found =
      scan_binary(&part, AGX_ISA_ENCODING_AGX3B, want, count);
   CHECK(found.fence_at[1] > found.last_at);
   k.size = part.info.main_size;
   printf("bytes=%u gprs=%u fences=%u/%u\n", k.size, part.info.nr_gprs,
          found.fences[0], found.fences[1]);
   free(part.binary);
   ralloc_free(b.shader);
   return k;
}

/* Kernels with several queries in a thread. */
static unsigned
thread_cases(void)
{
   const unsigned count = 2 + 2 + 1 + 19;
   set_arm(NULL, NULL);
   unsetenv(SKIP);
   const struct thread_kernel plain = thread_kernel("defaults");
   CHECK(plain.in_last == 0 && plain.elsewhere == count);

   setenv(SKIP, "0", 1);
   const struct thread_kernel every = thread_kernel("every thread sweeps");
   CHECK(every.in_last == count && every.elsewhere == 0);
   CHECK(every.size != plain.size);
   unsetenv(SKIP);

   /* Each launch site carries two more branches for the state a thread's
    * first launch stores: around the context copy and around the constant
    * words of the save area. */
   setenv(ONCE, "0", 1);
   const struct thread_kernel each = thread_kernel("state at every launch");
   CHECK(each.branches + 2 * 2 == plain.branches);
   CHECK(each.elsewhere == count);
   CHECK(each.size != plain.size);
   unsetenv(ONCE);

   /* One launch site, outside any loop: no thread can have launched before
    * it, so the branch folds away and the kernel is the same whether or not
    * the state is stored once. A site inside a loop keeps the branch. */
   struct scan found;
   const unsigned one = compile_kernel(0x43, NULL, NULL, 0, &found);
   setenv(ONCE, "0", 1);
   CHECK(compile_kernel(0x43, NULL, NULL, 0, &found) == one);
   unsetenv(ONCE);
   return 5;
}

int
main(void)
{
   setbuf(stdout, NULL);
   glsl_type_singleton_init_or_ref();
   const unsigned cases = lowering_cases() + backend_cases() + thread_cases();
   glsl_type_singleton_decref();
   printf("Gen1 SCS sweep: %u passed\n", cases);
   return 0;
}
