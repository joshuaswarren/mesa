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
   size_t base_offset; /* Simulated runtime VkDescriptorBufferInfo.offset for alignment checks */
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
         case nir_op_u2u16:
         case nir_op_i2i16:
            r = (uint16_t)s0;
            break;
         case nir_op_u2u8:
         case nir_op_i2i8:
            r = (uint8_t)s0;
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


struct hk_eval_shared_var {
   const nir_variable *var;
   size_t offset;
   size_t size;
};

static inline size_t
hk_eval_type_bytes(const struct glsl_type *type)
{
   if (glsl_type_is_array(type)) {
      unsigned len = glsl_get_length(type);
      uint32_t stride = hk_eval_type_stride(type);
      return (size_t)len * stride;
   }
   return (size_t)glsl_get_components(type) * (glsl_get_bit_size(type) / 8);
}

enum hk_wg_thr_state {
   HK_WG_RUN = 0,
   HK_WG_BARRIER,
   HK_WG_CMAT,
   HK_WG_DONE,
};

struct hk_wg_frame {
   const struct exec_list *list;
   nir_cf_node *node;
   nir_instr *next_instr;
   bool is_loop;
   unsigned iters;
   bool entered_block;
};

struct hk_wg_thread {
   struct hk_eval_env env;
   uint32_t sg_id;
   uint32_t sg_lane;
   nir_block *prev_block;
   struct hk_wg_frame stack[16];
   int sp;
   enum hk_wg_thr_state state;
   const nir_intrinsic_instr *pending_cmat;
};

static inline void
hk_wg_decode_addr(const struct hk_eval_env *env, uint8_t *shared_mem,
                  const struct hk_eval_shared_var *svars, unsigned num_svars,
                  uint64_t addr, uint8_t **ptr_out, size_t *avail_out,
                  size_t *align_base_out)
{
   uint64_t tag = addr >> 40;
   uint64_t off = addr & ((1ull << 40) - 1);
   if (tag >= 1 && tag <= env->num_ssbos) {
      const struct hk_eval_ssbo *buf = &env->ssbos[tag - 1];
      HK_EVAL_CHECK(off <= buf->size);
      *ptr_out = (uint8_t *)buf->data + off;
      *avail_out = buf->size - (size_t)off;
      *align_base_out = buf->base_offset + (size_t)off;
      return;
   }
   uint64_t sidx = tag - (env->num_ssbos + 1);
   HK_EVAL_CHECK(tag > env->num_ssbos && sidx < num_svars);
   HK_EVAL_CHECK(off <= svars[sidx].size);
   *ptr_out = shared_mem + svars[sidx].offset + off;
   *avail_out = svars[sidx].size - (size_t)off;
   *align_base_out = svars[sidx].offset + (size_t)off;
}

static inline void
hk_wg_step_thread(struct hk_wg_thread *thr, uint8_t *shared_mem,
                  const struct hk_eval_shared_var *svars, unsigned num_svars)
{
   struct hk_eval_env *env = &thr->env;
   while (thr->state == HK_WG_RUN) {
      if (thr->sp < 0) {
         thr->state = HK_WG_DONE;
         return;
      }
      struct hk_wg_frame *fr = &thr->stack[thr->sp];
      if (!fr->node) {
         if (fr->is_loop) {
            HK_EVAL_CHECK(++fr->iters < 100000);
            fr->node = exec_list_is_empty(fr->list)
                          ? NULL
                          : exec_node_data(nir_cf_node, exec_list_get_head_const(fr->list), node);
            fr->next_instr = NULL;
            fr->entered_block = false;
            continue;
         }
         thr->sp--;
         continue;
      }

      nir_cf_node *cf = fr->node;
      if (cf->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(cf);
         bool cond = (hk_eval_read_ssa(env, nif->condition.ssa, 0) & 1) != 0;
         fr->node = nir_cf_node_next(cf);
         fr->next_instr = NULL;
         fr->entered_block = false;
         const struct exec_list *arm = cond ? &nif->then_list : &nif->else_list;
         HK_EVAL_CHECK(thr->sp + 1 < 16);
         thr->sp++;
         thr->stack[thr->sp] = (struct hk_wg_frame){
            .list = arm,
            .node = exec_list_is_empty(arm)
                       ? NULL
                       : exec_node_data(nir_cf_node, exec_list_get_head_const(arm), node),
         };
         continue;
      }

      if (cf->type == nir_cf_node_loop) {
         nir_loop *lp = nir_cf_node_as_loop(cf);
         fr->node = nir_cf_node_next(cf);
         fr->next_instr = NULL;
         fr->entered_block = false;
         HK_EVAL_CHECK(thr->sp + 1 < 16);
         thr->sp++;
         thr->stack[thr->sp] = (struct hk_wg_frame){
            .list = &lp->body,
            .node = exec_list_is_empty(&lp->body)
                       ? NULL
                       : exec_node_data(nir_cf_node, exec_list_get_head_const(&lp->body), node),
            .is_loop = true,
         };
         continue;
      }

      HK_EVAL_CHECK(cf->type == nir_cf_node_block);
      nir_block *block = nir_cf_node_as_block(cf);
      if (!fr->entered_block) {
         fr->entered_block = true;
         /* Evaluate phis simultaneously using thr->prev_block. */
         unsigned phi_count = 0;
         nir_foreach_phi(phi, block)
            phi_count++;
         if (phi_count) {
            uint64_t (*phi_vals)[4] = calloc(phi_count, sizeof(*phi_vals));
            HK_EVAL_CHECK(phi_vals && thr->prev_block);
            unsigned idx = 0;
            nir_foreach_phi(phi, block) {
               nir_phi_src *src = nir_phi_get_src_from_block(phi, thr->prev_block);
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
         fr->next_instr = exec_list_is_empty(&block->instr_list)
                             ? NULL
                             : exec_node_data(nir_instr, exec_list_get_head(&block->instr_list), node);
      }

      if (!fr->next_instr) {
         thr->prev_block = block;
         fr->node = nir_cf_node_next(cf);
         fr->entered_block = false;
         continue;
      }

      nir_instr *instr = fr->next_instr;
      struct exec_node *nxt = exec_node_get_next(&instr->node);
      fr->next_instr = exec_node_is_tail_sentinel(nxt)
                          ? NULL
                          : exec_node_data(nir_instr, nxt, node);

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
         case nir_deref_type_var: {
            HK_EVAL_CHECK(d->var && d->var->data.mode == nir_var_mem_shared);
            unsigned sidx = num_svars;
            for (unsigned i = 0; i < num_svars; ++i) {
               if (svars[i].var == d->var) {
                  sidx = i;
                  break;
               }
            }
            HK_EVAL_CHECK(sidx < num_svars);
            addr = (((uint64_t)(env->num_ssbos + 1 + sidx)) << 40);
            break;
         }
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
            HK_EVAL_CHECK(!"unsupported deref type in wg eval");
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
         case nir_intrinsic_load_subgroup_id:
            env->ssa[intr->def.index][0] = thr->sg_id;
            break;
         case nir_intrinsic_load_subgroup_invocation:
            env->ssa[intr->def.index][0] = thr->sg_lane;
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
            uint8_t *ptr = NULL;
            size_t avail = 0, abase = 0;
            hk_wg_decode_addr(env, shared_mem, svars, num_svars, vaddr,
                              &ptr, &avail, &abase);
            if (intr->intrinsic == nir_intrinsic_load_global) {
               unsigned amul = nir_intrinsic_align_mul(intr);
               if (amul > 1)
                  HK_EVAL_CHECK((abase % amul) == 0);
            }
            const unsigned nbytes = intr->def.bit_size / 8;
            HK_EVAL_CHECK((size_t)intr->def.num_components * nbytes <= avail);
            for (unsigned c = 0; c < intr->def.num_components; ++c) {
               uint32_t v = 0;
               memcpy(&v, ptr + (size_t)c * nbytes, nbytes);
               env->ssa[intr->def.index][c] = v;
            }
            break;
         }
         case nir_intrinsic_store_deref:
         case nir_intrinsic_store_global: {
            const bool is_deref = intr->intrinsic == nir_intrinsic_store_deref;
            uint64_t vaddr = hk_eval_read_ssa(env, intr->src[is_deref ? 0 : 1].ssa, 0);
            nir_def *val = intr->src[is_deref ? 1 : 0].ssa;
            uint8_t *ptr = NULL;
            size_t avail = 0, abase = 0;
            hk_wg_decode_addr(env, shared_mem, svars, num_svars, vaddr,
                              &ptr, &avail, &abase);
            if (!is_deref) {
               unsigned amul = nir_intrinsic_align_mul(intr);
               if (amul > 1)
                  HK_EVAL_CHECK((abase % amul) == 0);
            }
            const unsigned nbytes = val->bit_size / 8;
            const unsigned wmask = nir_intrinsic_write_mask(intr);
            for (unsigned c = 0; c < val->num_components; ++c) {
               if (!(wmask & (1u << c)))
                  continue;
               HK_EVAL_CHECK((size_t)(c + 1) * nbytes <= avail);
               uint32_t v = (uint32_t)hk_eval_read_ssa(env, val, c);
               memcpy(ptr + (size_t)c * nbytes, &v, nbytes);
            }
            break;
         }
         case nir_intrinsic_barrier:
            if (nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP) {
               thr->state = HK_WG_BARRIER;
               return;
            }
            break;
         case nir_intrinsic_cmat_muladd_agx:
            thr->pending_cmat = intr;
            thr->state = HK_WG_CMAT;
            return;
         default:
            fprintf(stderr, "hk_nir_eval_test(wg): unsupported intrinsic %s\n",
                    nir_intrinsic_infos[intr->intrinsic].name);
            HK_EVAL_CHECK(!"unsupported intrinsic in wg eval");
         }
         break;
      }
      case nir_instr_type_jump: {
         nir_jump_instr *j = nir_instr_as_jump(instr);
         thr->prev_block = block;
         if (j->type == nir_jump_break) {
            while (thr->sp >= 0 && !thr->stack[thr->sp].is_loop)
               thr->sp--;
            HK_EVAL_CHECK(thr->sp >= 0);
            thr->sp--;
         } else if (j->type == nir_jump_continue) {
            while (thr->sp >= 0 && !thr->stack[thr->sp].is_loop)
               thr->sp--;
            HK_EVAL_CHECK(thr->sp >= 0);
            thr->stack[thr->sp].node = NULL;
            thr->stack[thr->sp].next_instr = NULL;
            thr->stack[thr->sp].entered_block = false;
         } else if (j->type == nir_jump_return) {
            thr->sp = -1;
            thr->state = HK_WG_DONE;
            return;
         }
         break;
      }
      default:
         HK_EVAL_CHECK(!"unsupported instr type in wg eval");
      }
   }
}

static inline void
hk_eval_dispatch_wg(nir_shader *nir, const uint32_t grid[3],
                    const struct hk_eval_ssbo *ssbos, unsigned num_ssbos,
                    const void *push, size_t push_size)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);

   const uint32_t lx = nir->info.workgroup_size[0];
   const uint32_t ly = nir->info.workgroup_size[1];
   const uint32_t lz = nir->info.workgroup_size[2];
   const uint32_t wg_threads = lx * ly * lz;
   HK_EVAL_CHECK(wg_threads > 0 && wg_threads <= 1024);

   struct hk_eval_shared_var svars[16];
   unsigned num_svars = 0;
   size_t total_shared = 0;
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_shared) {
      HK_EVAL_CHECK(num_svars < ARRAY_SIZE(svars));
      total_shared = (total_shared + 15u) & ~(size_t)15u;
      size_t sz = hk_eval_type_bytes(var->type);
      svars[num_svars++] = (struct hk_eval_shared_var){
         .var = var,
         .offset = total_shared,
         .size = sz,
      };
      total_shared += sz;
   }
   uint8_t *shared_mem = total_shared ? calloc(1, total_shared) : NULL;
   HK_EVAL_CHECK(!total_shared || shared_mem);

   struct hk_wg_thread *thrs = calloc(wg_threads, sizeof(*thrs));
   uint64_t (*all_ssa)[4] = calloc((size_t)wg_threads * (impl->ssa_alloc + 1),
                                   sizeof(*all_ssa));
   HK_EVAL_CHECK(thrs && all_ssa);

   for (uint32_t wz = 0; wz < grid[2]; ++wz) {
      for (uint32_t wy = 0; wy < grid[1]; ++wy) {
         for (uint32_t wx = 0; wx < grid[0]; ++wx) {
            if (total_shared)
               memset(shared_mem, 0, total_shared);
            memset(all_ssa, 0,
                   (size_t)wg_threads * (impl->ssa_alloc + 1) * sizeof(*all_ssa));

            for (uint32_t tid = 0; tid < wg_threads; ++tid) {
               uint32_t tx = tid % lx;
               uint32_t ty = (tid / lx) % ly;
               uint32_t tz = tid / (lx * ly);
               thrs[tid] = (struct hk_wg_thread){
                  .env = {
                     .ssbos = ssbos,
                     .num_ssbos = num_ssbos,
                     .push = push,
                     .push_size = push_size,
                     .wg_id = { wx, wy, wz },
                     .lid = { tx, ty, tz },
                     .gid = { wx * lx + tx, wy * ly + ty, wz * lz + tz },
                     .ssa = all_ssa + (size_t)tid * (impl->ssa_alloc + 1),
                     .num_ssa = impl->ssa_alloc + 1,
                  },
                  .sg_id = tid / 32u,
                  .sg_lane = tid % 32u,
                  .sp = 0,
                  .state = HK_WG_RUN,
               };
               thrs[tid].stack[0] = (struct hk_wg_frame){
                  .list = &impl->body,
                  .node = exec_list_is_empty(&impl->body)
                             ? NULL
                             : exec_node_data(nir_cf_node, exec_list_get_head_const(&impl->body), node),
               };
            }

            for (;;) {
               bool any_run = false;
               for (uint32_t tid = 0; tid < wg_threads; ++tid) {
                  if (thrs[tid].state == HK_WG_RUN) {
                     hk_wg_step_thread(&thrs[tid], shared_mem, svars, num_svars);
                     any_run = true;
                  }
               }

               /* Resolve any subgroup where all 32 lanes hit cmat_muladd_agx. */
               bool resolved_cmat = false;
               for (uint32_t sg = 0; sg * 32u < wg_threads; ++sg) {
                  uint32_t base = sg * 32u;
                  if (base + 32u > wg_threads)
                     break;
                  const nir_intrinsic_instr *cm = thrs[base].pending_cmat;
                  if (thrs[base].state != HK_WG_CMAT || !cm)
                     continue;
                  bool all_ready = true;
                  for (uint32_t l = 1; l < 32u; ++l) {
                     if (thrs[base + l].state != HK_WG_CMAT ||
                         thrs[base + l].pending_cmat != cm) {
                        all_ready = false;
                        break;
                     }
                  }
                  if (!all_ready)
                     continue;

                  /* AGX 8x8 FP32 tile layout across 32 subgroup lanes:
                   *   row = ((lane >> 1) & 3) | ((lane >> 2) & 4)
                   *   col0 = ((lane & 1) << 1) | ((lane >> 1) & 4), col1 = col0 + 1 */
                  float A[8][8], B[8][8], C[8][8];
                  for (uint32_t l = 0; l < 32u; ++l) {
                     uint32_t r = ((l >> 1) & 3u) | ((l >> 2) & 4u);
                     uint32_t c0 = ((l & 1u) << 1) | ((l >> 1) & 4u);
                     struct hk_eval_env *e = &thrs[base + l].env;
                     A[r][c0 + 0] = hk_eval_u2f((uint32_t)hk_eval_read_ssa(e, cm->src[0].ssa, 0));
                     A[r][c0 + 1] = hk_eval_u2f((uint32_t)hk_eval_read_ssa(e, cm->src[0].ssa, 1));
                     B[r][c0 + 0] = hk_eval_u2f((uint32_t)hk_eval_read_ssa(e, cm->src[1].ssa, 0));
                     B[r][c0 + 1] = hk_eval_u2f((uint32_t)hk_eval_read_ssa(e, cm->src[1].ssa, 1));
                     C[r][c0 + 0] = hk_eval_u2f((uint32_t)hk_eval_read_ssa(e, cm->src[2].ssa, 0));
                     C[r][c0 + 1] = hk_eval_u2f((uint32_t)hk_eval_read_ssa(e, cm->src[2].ssa, 1));
                  }
                  float D[8][8];
                  for (unsigned r = 0; r < 8; ++r) {
                     for (unsigned c = 0; c < 8; ++c) {
                        float acc = C[r][c];
                        for (unsigned k = 0; k < 8; ++k)
                           acc = fmaf(A[r][k], B[k][c], acc);
                        D[r][c] = acc;
                     }
                  }
                  for (uint32_t l = 0; l < 32u; ++l) {
                     uint32_t r = ((l >> 1) & 3u) | ((l >> 2) & 4u);
                     uint32_t c0 = ((l & 1u) << 1) | ((l >> 1) & 4u);
                     struct hk_eval_env *e = &thrs[base + l].env;
                     e->ssa[cm->def.index][0] = hk_eval_f2u(D[r][c0 + 0]);
                     e->ssa[cm->def.index][1] = hk_eval_f2u(D[r][c0 + 1]);
                     thrs[base + l].pending_cmat = NULL;
                     thrs[base + l].state = HK_WG_RUN;
                  }
                  resolved_cmat = true;
               }
               if (resolved_cmat)
                  continue;

               unsigned n_done = 0, n_bar = 0;
               for (uint32_t tid = 0; tid < wg_threads; ++tid) {
                  if (thrs[tid].state == HK_WG_DONE)
                     n_done++;
                  else if (thrs[tid].state == HK_WG_BARRIER)
                     n_bar++;
               }
               if (n_done == wg_threads)
                  break;
               if (n_bar == wg_threads) {
                  for (uint32_t tid = 0; tid < wg_threads; ++tid)
                     thrs[tid].state = HK_WG_RUN;
                  continue;
               }
               HK_EVAL_CHECK(any_run);
            }
         }
      }
   }

   free(all_ssa);
   free(thrs);
   free(shared_mem);
}

#endif
