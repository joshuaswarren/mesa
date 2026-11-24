/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#ifndef HK_NIR_EVAL_TEST_H
#define HK_NIR_EVAL_TEST_H

#include <stdio.h>
#include <stdlib.h>

#define HK_EVAL_CHECK(cond)                                                    \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,      \
                 #cond);                                                       \
         abort();                                                              \
      }                                                                        \
   } while (0)
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"

struct hk_eval_ssbo {
   unsigned set;
   unsigned binding;
   void *data;
   size_t size;
};

struct hk_eval_env {
   const struct hk_eval_ssbo *ssbos;
   unsigned num_ssbos;
   const void *push;
   size_t push_size;
   uint32_t wg_id[3];
   uint32_t lid[3];
   uint32_t gid[3];
   uint32_t base_wg[3];
   uint64_t (*ssa)[4];
   unsigned num_ssa;
};

static inline float
hk_eval_u2f(uint32_t u)
{
   float f;
   memcpy(&f, &u, sizeof(f));
   return f;
}

static inline uint32_t
hk_eval_f2u(float f)
{
   uint32_t u;
   memcpy(&u, &f, sizeof(u));
   return u;
}

static inline uint64_t
hk_eval_read_ssa(const struct hk_eval_env *env, nir_def *def, unsigned comp)
{
   HK_EVAL_CHECK(def && def->index < env->num_ssa && comp < 4);
   return env->ssa[def->index][comp];
}

static inline uint64_t
hk_eval_alu_src(const struct hk_eval_env *env, const nir_alu_src *src,
                unsigned comp)
{
   return hk_eval_read_ssa(env, src->src.ssa, src->swizzle[comp]);
}

static inline int
hk_eval_find_ssbo(const struct hk_eval_env *env, unsigned set, unsigned binding)
{
   for (unsigned i = 0; i < env->num_ssbos; ++i) {
      if (env->ssbos[i].set == set && env->ssbos[i].binding == binding)
         return (int)i;
   }
   HK_EVAL_CHECK(!"SSBO binding not found in hk_eval_env");
   return -1;
}

static inline uint64_t
hk_eval_make_vaddr(unsigned ssbo_idx, uint64_t offset)
{
   HK_EVAL_CHECK(offset < (1ull << 40));
   return (((uint64_t)(ssbo_idx + 1)) << 40) | offset;
}

static inline void
hk_eval_decode_vaddr(const struct hk_eval_env *env, uint64_t addr,
                     const struct hk_eval_ssbo **buf_out, uint64_t *off_out)
{
   uint64_t tag = addr >> 40;
   HK_EVAL_CHECK(tag >= 1 && tag <= env->num_ssbos);
   *buf_out = &env->ssbos[tag - 1];
   *off_out = addr & ((1ull << 40) - 1);
}

static inline uint32_t
hk_eval_type_stride(const struct glsl_type *type)
{
   unsigned exp = glsl_get_explicit_stride(type);
   if (exp)
      return exp;
   if (glsl_type_is_array(type)) {
      const struct glsl_type *elem = glsl_get_array_element(type);
      return glsl_get_components(elem) * (glsl_get_bit_size(elem) / 8);
   }
   return glsl_get_components(type) * (glsl_get_bit_size(type) / 8);
}

static inline void
hk_eval_alu(struct hk_eval_env *env, const nir_alu_instr *alu)
{
   const unsigned comps = alu->def.num_components;
   const unsigned bs = alu->def.bit_size;
   uint64_t res[4] = {0};

   switch (alu->op) {
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
      for (unsigned c = 0; c < comps; ++c)
         res[c] = hk_eval_alu_src(env, &alu->src[c], 0);
      break;
   default:
      for (unsigned c = 0; c < comps; ++c) {
         uint64_t s0 = nir_op_infos[alu->op].num_inputs > 0
                          ? hk_eval_alu_src(env, &alu->src[0], c)
                          : 0;
         uint64_t s1 = nir_op_infos[alu->op].num_inputs > 1
                          ? hk_eval_alu_src(env, &alu->src[1], c)
                          : 0;
         uint64_t s2 = nir_op_infos[alu->op].num_inputs > 2
                          ? hk_eval_alu_src(env, &alu->src[2], c)
                          : 0;
         uint64_t r = 0;
         switch (alu->op) {
         case nir_op_mov:
            r = s0;
            break;
         case nir_op_iadd:
            r = (bs == 64) ? (s0 + s1) : (uint32_t)((uint32_t)s0 + (uint32_t)s1);
            break;
         case nir_op_isub:
            r = (bs == 64) ? (s0 - s1) : (uint32_t)((uint32_t)s0 - (uint32_t)s1);
            break;
         case nir_op_imul:
            r = (bs == 64) ? (s0 * s1) : (uint32_t)((uint32_t)s0 * (uint32_t)s1);
            break;
         case nir_op_ineg:
            r = (bs == 64) ? (uint64_t)(-(int64_t)s0)
                           : (uint32_t)(-(int32_t)(uint32_t)s0);
            break;
         case nir_op_iabs: {
            int32_t v = (int32_t)(uint32_t)s0;
            r = (uint32_t)(v < 0 ? -v : v);
            break;
         }
         case nir_op_idiv: {
            int32_t a = (int32_t)(uint32_t)s0, b = (int32_t)(uint32_t)s1;
            HK_EVAL_CHECK(b != 0);
            r = (uint32_t)(a / b);
            break;
         }
         case nir_op_udiv: {
            uint32_t a = (uint32_t)s0, b = (uint32_t)s1;
            HK_EVAL_CHECK(b != 0);
            r = a / b;
            break;
         }
         case nir_op_irem: {
            int32_t a = (int32_t)(uint32_t)s0, b = (int32_t)(uint32_t)s1;
            HK_EVAL_CHECK(b != 0);
            r = (uint32_t)(a % b);
            break;
         }
         case nir_op_imod: {
            int32_t a = (int32_t)(uint32_t)s0, b = (int32_t)(uint32_t)s1;
            HK_EVAL_CHECK(b != 0);
            int32_t rem = a % b;
            if (rem != 0 && ((rem < 0) != (b < 0)))
               rem += b;
            r = (uint32_t)rem;
            break;
         }
         case nir_op_umod: {
            uint32_t a = (uint32_t)s0, b = (uint32_t)s1;
            HK_EVAL_CHECK(b != 0);
            r = a % b;
            break;
         }
         case nir_op_ishl:
            r = (bs == 64) ? (s0 << (s1 & 63)) : (uint32_t)((uint32_t)s0 << (s1 & 31));
            break;
         case nir_op_ishr:
            r = (bs == 64) ? (uint64_t)((int64_t)s0 >> (s1 & 63))
                           : (uint32_t)((int32_t)(uint32_t)s0 >> (s1 & 31));
            break;
         case nir_op_ushr:
            r = (bs == 64) ? (s0 >> (s1 & 63)) : ((uint32_t)s0 >> (s1 & 31));
            break;
         case nir_op_iand:
            r = (bs == 1) ? ((s0 & 1) & (s1 & 1)) : (s0 & s1);
            break;
         case nir_op_ior:
            r = (bs == 1) ? ((s0 & 1) | (s1 & 1)) : (s0 | s1);
            break;
         case nir_op_ixor:
            r = (bs == 1) ? ((s0 & 1) ^ (s1 & 1)) : (s0 ^ s1);
            break;
         case nir_op_inot:
            r = (bs == 1) ? (!(s0 & 1)) : (bs == 64 ? ~s0 : (uint32_t)(~(uint32_t)s0));
            break;
         case nir_op_ieq:
            r = (alu->src[0].src.ssa->bit_size == 64) ? (s0 == s1)
                                                      : ((uint32_t)s0 == (uint32_t)s1);
            break;
         case nir_op_ine:
            r = (alu->src[0].src.ssa->bit_size == 64) ? (s0 != s1)
                                                      : ((uint32_t)s0 != (uint32_t)s1);
            break;
         case nir_op_ilt:
            r = (int32_t)(uint32_t)s0 < (int32_t)(uint32_t)s1;
            break;
         case nir_op_ige:
            r = (int32_t)(uint32_t)s0 >= (int32_t)(uint32_t)s1;
            break;
         case nir_op_ult:
            r = (uint32_t)s0 < (uint32_t)s1;
            break;
         case nir_op_uge:
            r = (uint32_t)s0 >= (uint32_t)s1;
            break;
         case nir_op_imin:
            r = (uint32_t)MIN2((int32_t)(uint32_t)s0, (int32_t)(uint32_t)s1);
            break;
         case nir_op_imax:
            r = (uint32_t)MAX2((int32_t)(uint32_t)s0, (int32_t)(uint32_t)s1);
            break;
         case nir_op_umin:
            r = MIN2((uint32_t)s0, (uint32_t)s1);
            break;
         case nir_op_umax:
            r = MAX2((uint32_t)s0, (uint32_t)s1);
            break;
         case nir_op_bcsel:
            r = (s0 & 1) ? s1 : s2;
            break;
         case nir_op_b2i32:
            r = (s0 & 1) ? 1u : 0u;
            break;
         case nir_op_u2u64:
            r = (uint64_t)(uint32_t)s0;
            break;
         case nir_op_i2i64:
            r = (uint64_t)(int64_t)(int32_t)(uint32_t)s0;
            break;
         case nir_op_u2u32:
         case nir_op_i2i32:
            r = (uint32_t)s0;
            break;
         case nir_op_fadd:
            r = hk_eval_f2u(hk_eval_u2f((uint32_t)s0) + hk_eval_u2f((uint32_t)s1));
            break;
         case nir_op_fsub:
            r = hk_eval_f2u(hk_eval_u2f((uint32_t)s0) - hk_eval_u2f((uint32_t)s1));
            break;
         case nir_op_fmul:
            r = hk_eval_f2u(hk_eval_u2f((uint32_t)s0) * hk_eval_u2f((uint32_t)s1));
            break;
         case nir_op_ffma:
            r = hk_eval_f2u(fmaf(hk_eval_u2f((uint32_t)s0), hk_eval_u2f((uint32_t)s1),
                                 hk_eval_u2f((uint32_t)s2)));
            break;
         case nir_op_fneg:
            r = (uint32_t)s0 ^ 0x80000000u;
            break;
         case nir_op_fabs:
            r = (uint32_t)s0 & 0x7fffffffu;
            break;
         case nir_op_fmin: {
            float a = hk_eval_u2f((uint32_t)s0), b = hk_eval_u2f((uint32_t)s1);
            r = hk_eval_f2u(fminf(a, b));
            break;
         }
         case nir_op_fmax: {
            float a = hk_eval_u2f((uint32_t)s0), b = hk_eval_u2f((uint32_t)s1);
            r = hk_eval_f2u(fmaxf(a, b));
            break;
         }
         case nir_op_feq:
            r = hk_eval_u2f((uint32_t)s0) == hk_eval_u2f((uint32_t)s1);
            break;
         case nir_op_fneu:
            r = hk_eval_u2f((uint32_t)s0) != hk_eval_u2f((uint32_t)s1);
            break;
         case nir_op_flt:
            r = hk_eval_u2f((uint32_t)s0) < hk_eval_u2f((uint32_t)s1);
            break;
         case nir_op_fge:
            r = hk_eval_u2f((uint32_t)s0) >= hk_eval_u2f((uint32_t)s1);
            break;
         case nir_op_f2i32:
            r = (uint32_t)(int32_t)hk_eval_u2f((uint32_t)s0);
            break;
         case nir_op_f2u32:
            r = (uint32_t)hk_eval_u2f((uint32_t)s0);
            break;
         case nir_op_i2f32:
            r = hk_eval_f2u((float)(int32_t)(uint32_t)s0);
            break;
         case nir_op_u2f32:
            r = hk_eval_f2u((float)(uint32_t)s0);
            break;
         case nir_op_pack_64_2x32_split:
            r = ((uint64_t)(uint32_t)s1 << 32) | (uint32_t)s0;
            break;
         case nir_op_unpack_64_2x32_split_x:
            r = (uint32_t)s0;
            break;
         case nir_op_unpack_64_2x32_split_y:
            r = (uint32_t)(s0 >> 32);
            break;
         default:
            fprintf(stderr, "hk_nir_eval_test: unsupported alu op %s\n",
                    nir_op_infos[alu->op].name);
            HK_EVAL_CHECK(!"unsupported alu op");
         }
         res[c] = r;
      }
      break;
   }

   for (unsigned c = 0; c < comps; ++c)
      env->ssa[alu->def.index][c] = res[c];
}

enum hk_eval_flow {
   HK_EVAL_FLOW_NORMAL = 0,
   HK_EVAL_FLOW_BREAK,
   HK_EVAL_FLOW_CONTINUE,
   HK_EVAL_FLOW_RETURN,
};

static inline enum hk_eval_flow
hk_eval_cf_list(struct hk_eval_env *env, const struct exec_list *list,
                nir_block **prev_block);

static inline enum hk_eval_flow
hk_eval_block(struct hk_eval_env *env, nir_block *block, nir_block **prev_block)
{
   /* Evaluate all phis simultaneously using *prev_block. */
   unsigned phi_count = 0;
   nir_foreach_phi(phi, block)
      phi_count++;

   uint64_t (*phi_vals)[4] = NULL;
   if (phi_count) {
      phi_vals = calloc(phi_count, sizeof(*phi_vals));
      HK_EVAL_CHECK(phi_vals && *prev_block);
      unsigned idx = 0;
      nir_foreach_phi(phi, block) {
         nir_phi_src *src = nir_phi_get_src_from_block(phi, *prev_block);
         HK_EVAL_CHECK(src);
         for (unsigned c = 0; c < phi->def.num_components; ++c)
            phi_vals[idx][c] = hk_eval_read_ssa(env, src->src.ssa, c);
         idx++;
      }
      idx = 0;
      nir_foreach_phi(phi, block) {
         for (unsigned c = 0; c < phi->def.num_components; ++c)
            env->ssa[phi->def.index][c] = phi_vals[idx][c];
         idx++;
      }
      free(phi_vals);
   }

   enum hk_eval_flow flow = HK_EVAL_FLOW_NORMAL;
   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_phi:
         break;
      case nir_instr_type_load_const: {
         nir_load_const_instr *lc = nir_instr_as_load_const(instr);
         for (unsigned c = 0; c < lc->def.num_components; ++c) {
            if (lc->def.bit_size == 1)
               env->ssa[lc->def.index][c] = lc->value[c].b ? 1 : 0;
            else if (lc->def.bit_size == 64)
               env->ssa[lc->def.index][c] = lc->value[c].u64;
            else
               env->ssa[lc->def.index][c] = lc->value[c].u32;
         }
         break;
      }
      case nir_instr_type_undef: {
         nir_undef_instr *u = nir_instr_as_undef(instr);
         for (unsigned c = 0; c < u->def.num_components; ++c)
            env->ssa[u->def.index][c] = 0;
         break;
      }
      case nir_instr_type_alu:
         hk_eval_alu(env, nir_instr_as_alu(instr));
         break;
      case nir_instr_type_deref: {
         nir_deref_instr *d = nir_instr_as_deref(instr);
         uint64_t addr = 0;
         switch (d->deref_type) {
         case nir_deref_type_cast: {
            uint64_t parent = hk_eval_read_ssa(env, d->parent.ssa, 0);
            if (parent >> 40) {
               addr = parent;
            } else {
               unsigned set = (unsigned)(parent >> 16);
               unsigned binding = (unsigned)(parent & 0xffffu);
               int idx = hk_eval_find_ssbo(env, set, binding);
               addr = hk_eval_make_vaddr((unsigned)idx, 0);
            }
            break;
         }
         case nir_deref_type_struct: {
            nir_deref_instr *parent = nir_deref_instr_parent(d);
            uint64_t base = hk_eval_read_ssa(env, d->parent.ssa, 0);
            int off = glsl_get_struct_field_offset(parent->type, d->strct.index);
            addr = base + (uint64_t)(off > 0 ? off : 0);
            break;
         }
         case nir_deref_type_array: {
            nir_deref_instr *parent = nir_deref_instr_parent(d);
            uint64_t base = hk_eval_read_ssa(env, d->parent.ssa, 0);
            int64_t idx = (int64_t)(int32_t)(uint32_t)hk_eval_read_ssa(env, d->arr.index.ssa, 0);
            uint32_t stride = hk_eval_type_stride(parent->type);
            HK_EVAL_CHECK(idx >= 0);
            addr = base + (uint64_t)idx * stride;
            break;
         }
         default:
            HK_EVAL_CHECK(!"unsupported deref type");
         }
         env->ssa[d->def.index][0] = addr;
         break;
      }
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_global_invocation_id:
            for (unsigned c = 0; c < 3; ++c)
               env->ssa[intr->def.index][c] = env->gid[c];
            break;
         case nir_intrinsic_load_workgroup_id:
            for (unsigned c = 0; c < 3; ++c)
               env->ssa[intr->def.index][c] = env->wg_id[c];
            break;
         case nir_intrinsic_load_local_invocation_id:
            for (unsigned c = 0; c < 3; ++c)
               env->ssa[intr->def.index][c] = env->lid[c];
            break;
         case nir_intrinsic_load_base_workgroup_id:
            for (unsigned c = 0; c < 3; ++c)
               env->ssa[intr->def.index][c] = env->base_wg[c];
            break;
         case nir_intrinsic_load_push_constant: {
            uint64_t off = nir_intrinsic_base(intr) +
                           (uint32_t)hk_eval_read_ssa(env, intr->src[0].ssa, 0);
            const unsigned nbytes = intr->def.bit_size / 8;
            for (unsigned c = 0; c < intr->def.num_components; ++c) {
               uint64_t p = off + (uint64_t)c * nbytes;
               HK_EVAL_CHECK(p + nbytes <= env->push_size);
               uint32_t v = 0;
               memcpy(&v, (const uint8_t *)env->push + p, nbytes);
               env->ssa[intr->def.index][c] = v;
            }
            break;
         }
         case nir_intrinsic_vulkan_resource_index: {
            uint64_t key = ((uint64_t)nir_intrinsic_desc_set(intr) << 16) |
                           nir_intrinsic_binding(intr);
            for (unsigned c = 0; c < intr->def.num_components; ++c)
               env->ssa[intr->def.index][c] = key;
            break;
         }
         case nir_intrinsic_load_vulkan_descriptor:
            for (unsigned c = 0; c < intr->def.num_components; ++c)
               env->ssa[intr->def.index][c] = hk_eval_read_ssa(env, intr->src[0].ssa, c);
            break;
         case nir_intrinsic_deref_buffer_address:
            env->ssa[intr->def.index][0] = hk_eval_read_ssa(env, intr->src[0].ssa, 0);
            break;
         case nir_intrinsic_load_deref:
         case nir_intrinsic_load_global: {
            uint64_t vaddr = hk_eval_read_ssa(env, intr->src[0].ssa, 0);
            const struct hk_eval_ssbo *buf = NULL;
            uint64_t off = 0;
            hk_eval_decode_vaddr(env, vaddr, &buf, &off);
            const unsigned nbytes = intr->def.bit_size / 8;
            for (unsigned c = 0; c < intr->def.num_components; ++c) {
               uint64_t p = off + (uint64_t)c * nbytes;
               HK_EVAL_CHECK(p + nbytes <= buf->size);
               uint32_t v = 0;
               memcpy(&v, (const uint8_t *)buf->data + p, nbytes);
               env->ssa[intr->def.index][c] = v;
            }
            break;
         }
         case nir_intrinsic_store_deref:
         case nir_intrinsic_store_global: {
            const bool is_deref = intr->intrinsic == nir_intrinsic_store_deref;
            uint64_t vaddr = hk_eval_read_ssa(env, intr->src[is_deref ? 0 : 1].ssa, 0);
            nir_def *val = intr->src[is_deref ? 1 : 0].ssa;
            const struct hk_eval_ssbo *buf = NULL;
            uint64_t off = 0;
            hk_eval_decode_vaddr(env, vaddr, &buf, &off);
            const unsigned nbytes = val->bit_size / 8;
            const unsigned wmask = nir_intrinsic_write_mask(intr);
            for (unsigned c = 0; c < val->num_components; ++c) {
               if (!(wmask & (1u << c)))
                  continue;
               uint64_t p = off + (uint64_t)c * nbytes;
               HK_EVAL_CHECK(p + nbytes <= buf->size);
               uint32_t v = (uint32_t)hk_eval_read_ssa(env, val, c);
               memcpy((uint8_t *)buf->data + p, &v, nbytes);
            }
            break;
         }
         default:
            fprintf(stderr, "hk_nir_eval_test: unsupported intrinsic %s\n",
                    nir_intrinsic_infos[intr->intrinsic].name);
            HK_EVAL_CHECK(!"unsupported intrinsic");
         }
         break;
      }
      case nir_instr_type_jump: {
         nir_jump_instr *j = nir_instr_as_jump(instr);
         if (j->type == nir_jump_break)
            flow = HK_EVAL_FLOW_BREAK;
         else if (j->type == nir_jump_continue)
            flow = HK_EVAL_FLOW_CONTINUE;
         else if (j->type == nir_jump_return)
            flow = HK_EVAL_FLOW_RETURN;
         break;
      }
      default:
         HK_EVAL_CHECK(!"unsupported instr type");
      }
      if (flow != HK_EVAL_FLOW_NORMAL)
         break;
   }

   *prev_block = block;
   return flow;
}

static inline enum hk_eval_flow
hk_eval_cf_list(struct hk_eval_env *env, const struct exec_list *list,
                nir_block **prev_block)
{
   foreach_list_typed(nir_cf_node, cf, node, list) {
      if (cf->type == nir_cf_node_block) {
         enum hk_eval_flow f =
            hk_eval_block(env, nir_cf_node_as_block(cf), prev_block);
         if (f != HK_EVAL_FLOW_NORMAL)
            return f;
      } else if (cf->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(cf);
         bool cond = (hk_eval_read_ssa(env, nif->condition.ssa, 0) & 1) != 0;
         enum hk_eval_flow f = hk_eval_cf_list(
            env, cond ? &nif->then_list : &nif->else_list, prev_block);
         if (f != HK_EVAL_FLOW_NORMAL)
            return f;
      } else if (cf->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(cf);
         unsigned iters = 0;
         for (;;) {
            HK_EVAL_CHECK(++iters < 100000);
            enum hk_eval_flow f = hk_eval_cf_list(env, &loop->body, prev_block);
            if (f == HK_EVAL_FLOW_BREAK)
               break;
            if (f == HK_EVAL_FLOW_RETURN)
               return f;
         }
      } else {
         HK_EVAL_CHECK(!"unsupported cf node");
      }
   }
   return HK_EVAL_FLOW_NORMAL;
}

static inline void
hk_eval_dispatch(nir_shader *nir, const uint32_t grid[3],
                 const struct hk_eval_ssbo *ssbos, unsigned num_ssbos,
                 const void *push, size_t push_size)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   uint64_t (*ssa)[4] = calloc(impl->ssa_alloc + 1, sizeof(*ssa));
   HK_EVAL_CHECK(ssa);

   const uint32_t lx = nir->info.workgroup_size[0];
   const uint32_t ly = nir->info.workgroup_size[1];
   const uint32_t lz = nir->info.workgroup_size[2];

   struct hk_eval_env env = {
      .ssbos = ssbos,
      .num_ssbos = num_ssbos,
      .push = push,
      .push_size = push_size,
      .ssa = ssa,
      .num_ssa = impl->ssa_alloc + 1,
   };

   for (uint32_t wz = 0; wz < grid[2]; ++wz) {
      for (uint32_t wy = 0; wy < grid[1]; ++wy) {
         for (uint32_t wx = 0; wx < grid[0]; ++wx) {
            for (uint32_t tz = 0; tz < lz; ++tz) {
               for (uint32_t ty = 0; ty < ly; ++ty) {
                  for (uint32_t tx = 0; tx < lx; ++tx) {
                     env.wg_id[0] = wx;
                     env.wg_id[1] = wy;
                     env.wg_id[2] = wz;
                     env.lid[0] = tx;
                     env.lid[1] = ty;
                     env.lid[2] = tz;
                     env.gid[0] = wx * lx + tx;
                     env.gid[1] = wy * ly + ty;
                     env.gid[2] = wz * lz + tz;
                     nir_block *prev = NULL;
                     hk_eval_cf_list(&env, &impl->body, &prev);
                  }
               }
            }
         }
      }
   }
   free(ssa);
}

#endif
