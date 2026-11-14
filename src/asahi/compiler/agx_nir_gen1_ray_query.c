#include "agx_nir_gen1_ray_query.h"
#include "agx_nir.h"
#include "agx_nir_ray_query.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"
#include "util/u_debug.h"
#include <stdio.h>
#include <stdlib.h>

struct gen1_rq_state {
   nir_variable *flags;
   nir_variable *query;
   nir_variable *as, *origin, *direction, *tmin, *tmax, *mask, *active;
   nir_variable *result[4], *candidate[4], *started, *cursor, *refills;
};

static const struct glsl_type *
gen1_rq_field_type(const struct glsl_type *query, const struct glsl_type *field)
{
   if (!glsl_type_is_array(query))
      return field;
   return glsl_array_type(gen1_rq_field_type(glsl_get_array_element(query), field),
                          glsl_get_length(query), 0);
}

static nir_deref_instr *
gen1_rq_field_deref(nir_builder *b, nir_deref_instr *query, nir_variable *field)
{
   if (query->deref_type == nir_deref_type_var)
      return nir_build_deref_var(b, field);
   assert(query->deref_type == nir_deref_type_array);
   return nir_build_deref_follower(b,
      gen1_rq_field_deref(b, nir_deref_instr_parent(query), field), query);
}

nir_def *
agx_nir_gen1_vertex_position_address(nir_builder *b, nir_def *instance,
                                     nir_def *geometry, nir_def *primitive)
{
   nir_def *blas = nir_load_global(b, 1, 64, nir_iadd_imm(b, instance, 88), .align_mul = 8);
   nir_def *id = nir_load_global(b, 1, 32, nir_iadd_imm(b, blas, 176), .align_mul = 4);
   /* The sparse directory is indexed by the full 20-bit AS resource ID.
    * The former 11-bit mask aliased every later page onto the first page. */
   nir_def *entry = nir_iadd_imm(b, nir_u2u64(b, nir_imul_imm(b, id, 16)),
                                 AGX_GEN1_VERTEX_DIRECTORY_VA);
   nir_def *table = nir_load_global(b, 1, 64, entry, .align_mul = 8);
   nir_def *vertices = nir_load_global(b, 1, 32, nir_iadd_imm(b, table, 8), .align_mul = 4);
   nir_def *base = nir_load_global(b, 1, 32,
      nir_iadd(b, nir_iadd_imm(b, table, 16), nir_u2u64(b, nir_imul_imm(b, geometry, 4))), .align_mul = 4);
   nir_def *index = nir_iadd(b, base, primitive);
   return nir_iadd(b, nir_iadd(b, table, nir_u2u64(b, vertices)),
                   nir_u2u64(b, nir_imul_imm(b, index, 36)));
}

static bool
gen1_rq_supported_load(const nir_intrinsic_instr *intr, bool instances, bool position_fetch)
{
   switch (nir_intrinsic_ray_query_value(intr)) {
   case nir_ray_query_value_intersection_triangle_vertex_positions:
      return position_fetch && instances && nir_intrinsic_committed(intr);
   case nir_ray_query_value_flags:
   case nir_ray_query_value_tmin:
   case nir_ray_query_value_world_ray_origin:
   case nir_ray_query_value_world_ray_direction:
      return true;
   case nir_ray_query_value_intersection_instance_id:
   case nir_ray_query_value_intersection_instance_custom_index:
   case nir_ray_query_value_intersection_instance_address:
      return instances && nir_intrinsic_committed(intr);
   case nir_ray_query_value_intersection_type:
   case nir_ray_query_value_intersection_t:
   case nir_ray_query_value_intersection_primitive_index:
   case nir_ray_query_value_intersection_geometry_index:
   case nir_ray_query_value_intersection_front_face:
   case nir_ray_query_value_intersection_barycentrics:
      return nir_intrinsic_committed(intr);
   default:
      return false;
   }
}

static bool
gen1_candidate_supported_load(const nir_intrinsic_instr *intr, bool position_fetch)
{
   switch(nir_intrinsic_ray_query_value(intr)) {
   case nir_ray_query_value_intersection_triangle_vertex_positions:
      return position_fetch;
   case nir_ray_query_value_intersection_type:
   case nir_ray_query_value_intersection_t:
   case nir_ray_query_value_intersection_primitive_index:
   case nir_ray_query_value_intersection_geometry_index:
   case nir_ray_query_value_intersection_front_face:
   case nir_ray_query_value_intersection_barycentrics:
   case nir_ray_query_value_intersection_instance_id:
   case nir_ray_query_value_intersection_instance_custom_index:
   case nir_ray_query_value_intersection_instance_sbt_index:
   case nir_ray_query_value_intersection_candidate_aabb_opaque:
   case nir_ray_query_value_intersection_object_ray_origin:
   case nir_ray_query_value_intersection_object_ray_direction:
      return true;
   case nir_ray_query_value_intersection_object_to_world:
   case nir_ray_query_value_intersection_world_to_object:
      return nir_intrinsic_column(intr) < 4;
   default:return false;
   }
}

static void
gen1_candidate_packet(nir_builder *b,nir_def *query,nir_def *lane,
                      nir_def *cursor,nir_def *flags,nir_def *packet[4],
                      nir_def **opaque, nir_def **instance_flags)
{
   struct agx_gen1_ray_candidate c=agx_nir_gen1_load_candidate(b,query,lane,cursor);
   nir_def *leaf_type=nir_iand_imm(b,nir_ushr_imm(b,c.flags,11),7);
   nir_def *paired=nir_ieq_imm(b,leaf_type,1);
   nir_def *aabb=nir_ieq_imm(b,leaf_type,5);
   nir_def *geo=nir_load_global(b,1,32,nir_iadd(b,c.compressed_address,
      nir_u2u64(b,nir_bcsel(b,paired,nir_imm_int(b,48),nir_bcsel(b,aabb,nir_imm_int(b,24),nir_imm_int(b,12))))),.align_mul=4);
   nir_def *primoff=nir_bcsel(b,paired,nir_iadd_imm(b,nir_ishl_imm(b,
      nir_iand_imm(b,nir_ushr_imm(b,c.flags,10),1),2),52),nir_imm_int(b,28));
   nir_def *prim=nir_load_global(b,1,32,nir_iadd(b,c.compressed_address,
      nir_u2u64(b,primoff)),.align_mul=4);
   nir_def *node=agx_nir_gen1_candidate_instance_pointer(b,query,lane);
   nir_def *ids=nir_load_global(b,2,32,nir_iadd_imm(b,node,96),.align_mul=4);
   nir_def *marker=nir_load_global(b,1,32,nir_iadd_imm(b,c.compressed_address,-4),.align_mul=4);
   nir_def *iflags=nir_load_global(b,1,32,nir_iadd_imm(b,node,8),.align_mul=4);
   *instance_flags = iflags;
   nir_def *op=nir_bcsel(b,nir_test_mask(b,iflags,8),nir_imm_false(b),nir_bcsel(b,aabb,nir_test_mask(b,geo,0x80000000),nir_ieq_imm(b,nir_iand_imm(b,marker,31),8)));
   op=nir_ior(b,op,nir_test_mask(b,iflags,4));
   *opaque=nir_bcsel(b,nir_test_mask(b,flags,2),nir_imm_false(b),nir_ior(b,op,nir_test_mask(b,flags,1)));
   nir_def *zero=nir_imm_int(b,0);
   /* Shared native instance records keep world-to-object at16 and a
    * relative pointer to object-to-world/SBT metadata at104. Retain the
    * candidate's instance pointer when application code commits it. */
   packet[0]=nir_vec4(b,nir_unpack_64_2x32_split_x(b,node),
      nir_unpack_64_2x32_split_y(b,node),nir_b2i32(b,*opaque),zero);
   /* AABB queue records carry their entry distance in union word0,
    * not the triangle distance at+16. Native intersection_query.next uses
    * that entry value when pruning a candidate against a committed hit. */
   packet[1]=nir_vec4(b,zero,zero,
      nir_bcsel(b,aabb,nir_channel(b,c.barycentrics,0),c.distance),zero);
   packet[2]=nir_vec4(b,nir_ior(b,nir_iand_imm(b,prim,0x3fffffff),nir_bcsel(b,aabb,nir_imm_int(b,0x80000000),nir_imm_int(b,0x40000000))),
      nir_ior(b,nir_iand_imm(b,geo,0xffffff),nir_ishl_imm(b,nir_iand_imm(b,c.flags,1u<<8),23)),
      nir_channel(b,ids,1),nir_iand_imm(b,nir_channel(b,ids,0),0xffffff));
   packet[3]=nir_vec4(b,nir_channel(b,c.barycentrics,0),nir_channel(b,c.barycentrics,1),zero,zero);
}

/* Ray flags under which software must see every triangle hit: NoOpaque, the
 * facing and opacity culls, and SkipTriangles. The hardware is not given the
 * ray flags, so it may commit only when none of them is set. */
#define GEN1_RQ_SOFTWARE_TRIANGLE_FLAGS 0x1f2

/* Candidate control words: what a triangle hit does. QUEUE hands every hit
 * to the software candidate loop, whatever the geometry says. COMMIT_ALL has
 * the hardware commit every hit itself, which is all an Opaque ray needs.
 * COMMIT_OPAQUE leaves it to the geometry and the instance: the hardware
 * commits hits on opaque triangles and queues the others, and an instance's
 * force-opaque or force-no-opaque flag decides for the whole instance.
 *
 * A hardware commit lowers the tmax in payload word +24 to the hit distance,
 * keeps the hit words in query words 256/512, and keeps the hit's instance
 * pointer in the halfwords just before the current instance's (8960 rather
 * than 9152). The current instance is wherever traversal stopped, which can
 * be past the hit. A queued hit sends traversal back to the calling shader
 * once per leaf, so a query whose hits are committed takes about half the
 * GPU time. */
#define GEN1_RQ_CONTROL_QUEUE          0x9142
#define GEN1_RQ_CONTROL_COMMIT_OPAQUE  0x0142
#define GEN1_RQ_CONTROL_COMMIT_ALL     0x4942

static nir_def *
gen1_rq_committed_instance_pointer(nir_builder *b, nir_def *query, nir_def *lane)
{
   nir_def *at = nir_imul_imm(b, lane, 3), *half[3];
   for (unsigned i = 0; i < 3; ++i)
      half[i] = nir_u2u32(b, nir_load_scs_indexed_agx(b, 1, 16, query, at,
                                                     .base = 8960 + 2 * i, .component = 1));
   return nir_pack_64_2x32_split(b,
      nir_ior(b, half[0], nir_ishl_imm(b, half[1], 16)), half[2]);
}

static nir_def *
gen1_rq_hw_commits(nir_builder *b, nir_deref_instr *flags)
{
   return nir_inot(b, nir_test_mask(b, nir_load_deref(b, flags),
                                    GEN1_RQ_SOFTWARE_TRIANGLE_FLAGS));
}

static nir_def *
gen1_rq_control(nir_builder *b, nir_deref_instr *flags)
{
   nir_def *opaque = nir_test_mask(b, nir_load_deref(b, flags), 1);
   nir_def *commit = nir_bcsel(b, opaque,
                               nir_imm_int(b, GEN1_RQ_CONTROL_COMMIT_ALL),
                               nir_imm_int(b, GEN1_RQ_CONTROL_COMMIT_OPAQUE));
   return nir_bcsel(b, gen1_rq_hw_commits(b, flags), commit,
                    nir_imm_int(b, GEN1_RQ_CONTROL_QUEUE));
}

/* Committed values beyond the hit words and distance that a merged hardware
 * commit has to look up, so the merge skips what the shader never reads. */
enum gen1_rq_hw_fields {
   GEN1_RQ_HW_NODE = 1 << 0, /* the instance node pointer */
   GEN1_RQ_HW_IDS = 1 << 1,  /* instance ID and custom index, behind the node */
   GEN1_RQ_HW_UV = 1 << 2,   /* barycentrics */
};

static unsigned
gen1_rq_hw_fields(const nir_intrinsic_instr *intr)
{
   if (intr->intrinsic != nir_intrinsic_rq_load || !nir_intrinsic_committed(intr))
      return 0;
   switch (nir_intrinsic_ray_query_value(intr)) {
   case nir_ray_query_value_intersection_instance_id:
   case nir_ray_query_value_intersection_instance_custom_index:
      return GEN1_RQ_HW_NODE | GEN1_RQ_HW_IDS;
   case nir_ray_query_value_intersection_barycentrics:
      return GEN1_RQ_HW_UV;
   case nir_ray_query_value_intersection_instance_address:
   case nir_ray_query_value_intersection_instance_sbt_index:
   case nir_ray_query_value_intersection_object_ray_direction:
   case nir_ray_query_value_intersection_object_ray_origin:
   case nir_ray_query_value_intersection_object_to_world:
   case nir_ray_query_value_intersection_world_to_object:
   case nir_ray_query_value_intersection_triangle_vertex_positions:
      return GEN1_RQ_HW_NODE;
   default:
      return 0;
   }
}

/* After each trace or refill, take the hardware's committed hit when it is
 * closer than the committed one, so candidate pruning and confirmation compare
 * against the closest hit so far. Traversal clears query word 256 first, so a
 * hit tag there is this query's. With TerminateOnFirstHit the merged hit ends
 * the query, as a software commit of an opaque triangle does. */
static void
gen1_rq_merge_hw_commit(nir_builder *b, nir_def *id, nir_def *query,
                        nir_def *lane, nir_deref_instr *flags,
                        nir_deref_instr *active,
                        nir_deref_instr *result[4], unsigned fields)
{
   nir_push_if(b, gen1_rq_hw_commits(b, flags));
   nir_def *distance = nir_load_scs_uniform_agx(b, 1, 32, nir_load_preamble(b, 1, 64, .base = 118),
      nir_ishl_imm(b, id, 5), .base = 24, .format = PIPE_FORMAT_R32_UINT, .component = 0);
   nir_def *hit = nir_load_scs_indexed_agx(b, 2, 32, query, lane, .base = 256, .component = 3);
   nir_def *closer = nir_flt(b, distance, nir_channel(b, nir_load_deref(b, result[1]), 2));
   nir_push_if(b, nir_iand(b, nir_ine_imm(b, nir_ushr_imm(b, nir_channel(b, hit, 0), 30), 0), closer));
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *node = (fields & GEN1_RQ_HW_NODE) ?
      gen1_rq_committed_instance_pointer(b, query, lane) : nir_imm_int64(b, 0);
   nir_def *ids = (fields & GEN1_RQ_HW_IDS) ?
      nir_load_global(b, 2, 32, nir_iadd_imm(b, node, 96), .align_mul = 4) : nir_imm_ivec2(b, 0, 0);
   nir_def *uv = (fields & GEN1_RQ_HW_UV) ?
      nir_load_scs_indexed_agx(b, 2, 32, query, lane, .base = 512, .component = 3) : nir_imm_ivec2(b, 0, 0);
   /* The same packets a software commit of this hit would store. */
   nir_store_deref(b, result[0], nir_vec4(b, nir_unpack_64_2x32_split_x(b, node),
                                          nir_unpack_64_2x32_split_y(b, node),
                                          nir_imm_int(b, 1), zero), 15);
   nir_store_deref(b, result[1], nir_vec4(b, zero, zero, distance, zero), 15);
   nir_store_deref(b, result[2], nir_vec4(b, nir_channel(b, hit, 0), nir_channel(b, hit, 1),
                                          nir_channel(b, ids, 1),
                                          nir_iand_imm(b, nir_channel(b, ids, 0), 0xffffff)), 15);
   nir_store_deref(b, result[3], nir_vec4(b, nir_channel(b, uv, 0), nir_channel(b, uv, 1),
                                          zero, zero), 15);
   nir_push_if(b, nir_test_mask(b, nir_load_deref(b, flags), 4));
   nir_store_deref(b, active, nir_imm_false(b), 1);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

static void
gen1_candidate_next(nir_builder *b,nir_deref_instr *active,nir_deref_instr *cursor,
                     nir_deref_instr *flags,nir_deref_instr *minimum,
                     nir_deref_instr *result[4],nir_deref_instr *candidate[4],
                     nir_deref_instr *refills,nir_variable *more,
                     unsigned hw_fields)
{
   nir_def *id=nir_load_scs_thread_index_agx(b),*lane=nir_iand_imm(b,id,31);
   nir_def *query=nir_iadd_imm(b,nir_imul_imm(b,nir_ushr_imm(b,id,5),304*32),0x2f8000);
   nir_push_loop(b);
   nir_def *index=nir_load_deref(b,cursor);
   nir_def *control=nir_load_scs_indexed_agx(b,1,32,query,lane,.base=1792,.component=2);
   nir_def *count=nir_iand_imm(b,nir_ushr_imm(b,control,21),15);
   nir_push_if(b,nir_ult(b,index,count));
   nir_def *packet[4],*opaque,*instance_flags,*ray_flags=nir_load_deref(b,flags);
   gen1_candidate_packet(b,query,lane,index,ray_flags,packet,&opaque,&instance_flags);
   nir_store_deref(b,cursor,nir_iadd_imm(b,index,1),1);
   nir_def *distance=nir_channel(b,packet[1],2),*best=nir_channel(b,nir_load_deref(b,result[1]),2);
   nir_def *triangle=nir_ieq_imm(b,nir_ushr_imm(b,nir_channel(b,packet[2],0),30),1);
   /* Native box testing already establishes overlap with the initial ray
    * interval. Its entry may lie before tmin while its exit lies after it. */
   nir_def *interval=nir_iand(b,nir_ior(b,nir_inot(b,triangle),
      nir_fge(b,distance,nir_load_deref(b,minimum))),nir_fge(b,best,distance));
   nir_def *culled=nir_bcsel(b,opaque,nir_test_mask(b,ray_flags,0x40),nir_test_mask(b,ray_flags,0x80));
   nir_def *front = nir_ilt_imm(b, nir_channel(b, packet[2], 1), 0);
   nir_def *facing_culled = nir_bcsel(b, front,
      nir_test_mask(b, ray_flags, 0x20), nir_test_mask(b, ray_flags, 0x10));
   culled = nir_ior(b, culled, nir_iand(b, triangle,
      nir_iand(b, facing_culled, nir_inot(b, nir_test_mask(b, instance_flags, 1)))));
   nir_push_if(b,nir_iand(b,interval,nir_inot(b,culled)));

   nir_def *skip=nir_bcsel(b,triangle,nir_test_mask(b,ray_flags,256),nir_test_mask(b,ray_flags,512));
   nir_push_if(b,nir_inot(b,skip));
   nir_push_if(b,nir_iand(b,opaque,triangle));
   for(unsigned i=0;i<4;i++)nir_store_deref(b,result[i],packet[i],15);
   nir_push_if(b,nir_test_mask(b,ray_flags,4));
   nir_store_deref(b,active,nir_imm_false(b),1);nir_jump(b,nir_jump_break);
   nir_pop_if(b,NULL);
   nir_push_else(b,NULL);
   for(unsigned i=0;i<4;i++)nir_store_deref(b,candidate[i],packet[i],15);
   nir_store_var(b,more,nir_imm_true(b),1);nir_jump(b,nir_jump_break);
   nir_pop_if(b,NULL);nir_pop_if(b,NULL);nir_pop_if(b,NULL);
   nir_push_else(b,NULL);
   nir_def *refill=nir_iadd_imm(b,nir_load_deref(b,refills),1);
   nir_store_deref(b,refills,refill,1);
   nir_push_if(b,nir_ior(b,nir_ieq_imm(b,nir_iand_imm(b,control,1),0),
      nir_ugt_imm(b,refill,AGX_GEN1_CANDIDATE_REFILL_BOUND)));
   nir_store_deref(b,active,nir_imm_false(b),1);nir_jump(b,nir_jump_break);
   nir_pop_if(b,NULL);
   nir_def *completed=agx_nir_gen1_refill_candidate_queue(b,query,lane,id,nir_channel(b,nir_load_deref(b,result[1]),2));
   nir_store_deref(b,active,completed,1);
   gen1_rq_merge_hw_commit(b,id,query,lane,flags,active,result,hw_fields);
   nir_break_if(b,nir_inot(b,nir_load_deref(b,active)));
   nir_store_deref(b,cursor,nir_imm_int(b,0),1);
   nir_pop_if(b,NULL);nir_pop_loop(b,NULL);
}

static bool
gen1_rq_alias_copies(nir_function_impl *impl)
{
   bool progress = false;
   for (bool again = true; again;) {
      again = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
            if (store->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_deref_instr *dst = nir_src_as_deref(store->src[0]);
            nir_intrinsic_instr *load = nir_src_as_intrinsic(store->src[1]);
            if (dst->deref_type != nir_deref_type_var ||
                !dst->var->data.ray_query || !load ||
                load->intrinsic != nir_intrinsic_load_deref)
               continue;
            nir_deref_instr *src = nir_src_as_deref(load->src[0]);
            if (src->deref_type != nir_deref_type_var ||
                !src->var->data.ray_query)
               continue;
            if (src->var == dst->var) {
               nir_instr_remove(&store->instr);
               progress = again = true;
               continue;
            }
            unsigned stores = 0;
            nir_foreach_block(b2, impl) {
               nir_foreach_instr(i2, b2) {
                  if (i2->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *s2 = nir_instr_as_intrinsic(i2);
                  if (s2->intrinsic != nir_intrinsic_store_deref &&
                      s2->intrinsic != nir_intrinsic_copy_deref)
                     continue;
                  nir_deref_instr *d2 = nir_src_as_deref(s2->src[0]);
                  if (nir_deref_instr_get_variable(d2) == dst->var)
                     stores++;
               }
            }
            if (stores != 1 || dst->var->data.mode != nir_var_function_temp)
               continue;
            nir_variable *alias = dst->var, *target = src->var;
            nir_foreach_block(b2, impl) {
               nir_foreach_instr_safe(i2, b2) {
                  if (i2->type != nir_instr_type_deref)
                     continue;
                  nir_deref_instr *d = nir_instr_as_deref(i2);
                  if (d->deref_type != nir_deref_type_var || d->var != alias)
                     continue;
                  d->var = target;
                  d->modes = target->data.mode;
               }
            }
            nir_instr_remove(&store->instr);
            progress = again = true;
         }
      }
   }
   if (progress)
      nir_progress(true, impl, nir_metadata_control_flow);
   return progress;
}

/* A candidate-layout compute kernel ends with destination-less SCS loads over
 * storage its warp used: once per thread that launched a query, after
 * everything else the kernel does, between a traversal barrier of the scope
 * used ahead of a launch and one of the scope a traversal program ends with.
 * Every such lane issues every load. A row is what a warp reaches with four
 * words a lane: 512 bytes. The generated traversal programs end with the same
 * kind of load.
 *
 * By default the kernel names the per-thread storage a query of its warp
 * used: the payload records and the query block it stored to itself, and the
 * warp's records in the two other per-thread areas below the query blocks.
 * Without that, each ray query costs a fixed amount more on the GPU, the same
 * for cheap and for expensive rays; what the access does to the storage it
 * names is not established beyond that measurement.
 *
 * ASAHI_G17P_RQ_SCS_SWEEP=<mask> overrides the selection (0 ends the kernel
 * as before):
 *
 *   bit 0  the warp's query block, as far as its last instance pointer
 *   bit 1  the warp's payload records
 *   bit 2  the warp's saved context
 *   bit 3  the query context, which every thread of the dispatch shares
 *   bit 4  address the rows through a per-lane base register and no index,
 *          as the traversal programs address theirs, not base plus lane
 *   bit 5  the warp's records of other per-thread storage, named by
 *          ASAHI_G17P_RQ_SCS_SWEEP_EXTRA=<address>/<bytes>[,...]: <bytes> a
 *          thread from SCS offset <address>, below the query blocks
 *   bit 6  the warp's records in the two other per-thread areas:
 *          GEN1_RQ_LEVEL_PAYLOADS and GEN1_RQ_THREAD_RECORDS
 *
 * The loads come in the order payload, extra, the two areas, saved context,
 * query context, query block.
 */
enum gen1_rq_sweep {
   GEN1_RQ_SWEEP_QUERY = 1 << 0,
   GEN1_RQ_SWEEP_PAYLOAD = 1 << 1,
   GEN1_RQ_SWEEP_SAVED = 1 << 2,
   GEN1_RQ_SWEEP_CONTEXT = 1 << 3,
   GEN1_RQ_SWEEP_NO_INDEX = 1 << 4,
   GEN1_RQ_SWEEP_EXTRA = 1 << 5,
   GEN1_RQ_SWEEP_AREAS = 1 << 6,

   GEN1_RQ_SWEEP_DEFAULT = GEN1_RQ_SWEEP_QUERY | GEN1_RQ_SWEEP_PAYLOAD |
                           GEN1_RQ_SWEEP_AREAS,
};

/* The candidate layout: 96 warps, their query blocks from this offset on. */
#define GEN1_RQ_CANDIDATE_THREADS (96 * 32)
#define GEN1_RQ_QUERY_BLOCKS      0x2f8000

/* The candidate query block ends with a six-byte instance pointer a lane. */
#define GEN1_RQ_QUERY_USED_BYTES (9152 + 32 * 6)
#define GEN1_RQ_SWEEP_ROW_BYTES  (32 * 16)
/* Two more per-thread areas of the SCS, below the query blocks. The payload
 * section holds two runs of payload records 0x20000 apart, thirty-two bytes a
 * thread each; the kernel stores to the first and this is the second. The
 * section behind it starts with sixteen bytes a thread. */
#define GEN1_RQ_LEVEL_PAYLOADS       0x28000
#define GEN1_RQ_LEVEL_PAYLOAD_BYTES  32
#define GEN1_RQ_THREAD_RECORDS       0x48000
#define GEN1_RQ_THREAD_RECORD_BYTES  16
/* Loads the mask can select (19 + 2 + 3 + 7 + 3), and extra rows on top. */
#define GEN1_RQ_SWEEP_MASK_MAX   34
#define GEN1_RQ_SWEEP_EXTRA_MAX  32

struct gen1_rq_touch {
   nir_def *base, *index;
   uint16_t offset;
   uint8_t words, shift;
};

struct gen1_rq_sweep_state {
   nir_builder *b;
   nir_def *lane;
   bool no_index;
   unsigned count;
   struct gen1_rq_touch touch[GEN1_RQ_SWEEP_MASK_MAX + GEN1_RQ_SWEEP_EXTRA_MAX];
};

static void
gen1_rq_sweep_add(struct gen1_rq_sweep_state *s, nir_def *base, nir_def *index,
                  unsigned offset, unsigned words, unsigned shift)
{
   assert(s->count < ARRAY_SIZE(s->touch));
   s->touch[s->count++] = (struct gen1_rq_touch){
      .base = base, .index = index, .offset = offset, .words = words,
      .shift = shift,
   };
}

/* Cover `bytes` of a warp's own storage from `base`: whole rows, then a
 * remainder of one word a lane. */
static void
gen1_rq_sweep_rows(struct gen1_rq_sweep_state *s, nir_def *base, unsigned bytes)
{
   nir_builder *b = s->b;
   const unsigned rows = bytes / GEN1_RQ_SWEEP_ROW_BYTES;
   const unsigned rest = bytes % GEN1_RQ_SWEEP_ROW_BYTES;
   assert(rest == 0 || rest == 32 * 4);

   if (rows) {
      nir_def *row = s->no_index
         ? nir_iadd(b, base, nir_ishl_imm(b, s->lane, 4)) : base;
      for (unsigned i = 0; i < rows; ++i)
         gen1_rq_sweep_add(s, row, s->no_index ? NULL : s->lane,
                           GEN1_RQ_SWEEP_ROW_BYTES * i, 4, 4);
   }
   if (rest) {
      nir_def *row = s->no_index
         ? nir_iadd(b, base, nir_ishl_imm(b, s->lane, 2)) : base;
      gen1_rq_sweep_add(s, row, s->no_index ? NULL : s->lane,
                        GEN1_RQ_SWEEP_ROW_BYTES * rows, 1, 2);
   }
}

struct gen1_rq_sweep_extra {
   uint32_t address, bytes;
};

/* Parse the extra per-thread storage list. Anything malformed, storage that
 * does not end below the query blocks, or more rows than `max_rows`, drops
 * the whole list: a sweep over storage nobody named is worse than none. */
static unsigned
gen1_rq_sweep_extras(struct gen1_rq_sweep_extra *extra, unsigned max,
                     unsigned max_rows)
{
   const char *list = getenv("ASAHI_G17P_RQ_SCS_SWEEP_EXTRA");
   unsigned count = 0, rows = 0;
   if (!list || !*list)
      return 0;

   for (const char *at = list;;) {
      char *end;
      const unsigned long long address = strtoull(at, &end, 0);
      if (end == at || *end != '/')
         break;
      at = end + 1;
      const unsigned long long bytes = strtoull(at, &end, 0);
      /* Word-aligned storage, in whole rows or with a one-word remainder,
       * at offsets the access can carry. */
      if (end == at || (*end && *end != ',') || count == max ||
          (address & 3) || !bytes || bytes > 256 ||
          ((bytes % 16) && (bytes % 16) != 4) ||
          address > GEN1_RQ_QUERY_BLOCKS ||
          bytes * GEN1_RQ_CANDIDATE_THREADS > GEN1_RQ_QUERY_BLOCKS - address)
         break;
      extra[count++] = (struct gen1_rq_sweep_extra){address, bytes};
      rows += DIV_ROUND_UP(32 * bytes, GEN1_RQ_SWEEP_ROW_BYTES);
      if (rows > max_rows)
         break;
      if (!*end)
         return count;
      at = end + 1;
   }

   fprintf(stderr, "ASAHI_G17P_RQ_SCS_SWEEP_EXTRA: ignoring \"%s\"\n", list);
   return 0;
}

static void
gen1_rq_scs_sweep(nir_builder *b,
                  const struct agx_gen1_ray_query_layout *layout, unsigned mask)
{
   struct gen1_rq_sweep_state s = {
      .b = b, .no_index = mask & GEN1_RQ_SWEEP_NO_INDEX,
   };
   nir_def *id = nir_load_scs_thread_index_agx(b);
   nir_def *warp = nir_ushr_imm(b, id, 5);
   /* The warp's first thread: per-thread records start there. */
   nir_def *first = nir_iand_imm(b, id, ~31u);
   s.lane = nir_iand_imm(b, id, 31);

   if (mask & GEN1_RQ_SWEEP_PAYLOAD) {
      /* Thirty-two bytes a thread. */
      nir_def *payload = nir_iadd(b,
         nir_load_preamble(b, 1, 32, .base = layout->payload_uniform),
         nir_ishl_imm(b, first, 5));
      gen1_rq_sweep_rows(&s, payload, 32 * 32);
   }

   struct gen1_rq_sweep_extra extra[8];
   const unsigned extras = !(mask & GEN1_RQ_SWEEP_EXTRA) ? 0 :
      gen1_rq_sweep_extras(extra, ARRAY_SIZE(extra), GEN1_RQ_SWEEP_EXTRA_MAX);
   for (unsigned i = 0; i < extras; ++i)
      gen1_rq_sweep_rows(&s,
         nir_iadd_imm(b, nir_imul_imm(b, first, extra[i].bytes),
                      extra[i].address), 32 * extra[i].bytes);

   if (mask & GEN1_RQ_SWEEP_AREAS) {
      gen1_rq_sweep_rows(&s,
         nir_iadd_imm(b, nir_imul_imm(b, first, GEN1_RQ_LEVEL_PAYLOAD_BYTES),
                      GEN1_RQ_LEVEL_PAYLOADS),
         32 * GEN1_RQ_LEVEL_PAYLOAD_BYTES);
      gen1_rq_sweep_rows(&s,
         nir_iadd_imm(b, nir_imul_imm(b, first, GEN1_RQ_THREAD_RECORD_BYTES),
                      GEN1_RQ_THREAD_RECORDS),
         32 * GEN1_RQ_THREAD_RECORD_BYTES);
   }

   if (mask & GEN1_RQ_SWEEP_SAVED) {
      /* The three stores the launch makes, word for word. */
      nir_def *saved = nir_iadd(b, nir_load_preamble(b, 1, 32, .base = 116),
                                nir_imul_imm(b, warp, 48));
      gen1_rq_sweep_add(&s, saved, NULL, 256, 4, 0);
      gen1_rq_sweep_add(&s, saved, NULL, 272, 4, 0);
      gen1_rq_sweep_add(&s, saved, NULL, 288, 2, 0);
   }
   if (mask & GEN1_RQ_SWEEP_CONTEXT) {
      /* Likewise the seven stores of the context copy. */
      nir_def *context = nir_imm_int(b, 0x6000);
      for (unsigned i = 0; i < 6; ++i)
         gen1_rq_sweep_add(&s, context, NULL, 16 * i, 4, 0);
      gen1_rq_sweep_add(&s, context, NULL, 104, 2, 0);
   }
   if (mask & GEN1_RQ_SWEEP_QUERY)
      gen1_rq_sweep_rows(&s,
         nir_iadd_imm(b, nir_imul_imm(b, warp, 304 * 32),
                      GEN1_RQ_QUERY_BLOCKS),
         GEN1_RQ_QUERY_USED_BYTES);

   for (unsigned i = 0; i < s.count; ++i) {
      const struct gen1_rq_touch *t = &s.touch[i];
      const unsigned flags = AGX_NIR_SCS_TOUCH |
         (i == 0 ? AGX_NIR_SCS_TOUCH_FENCE_BEFORE : 0) |
         (i + 1 == s.count ? AGX_NIR_SCS_TOUCH_FENCE_AFTER : 0);
      if (t->index)
         nir_load_scs_indexed_agx(b, t->words, 32, t->base, t->index,
                                  .base = t->offset, .component = t->shift,
                                  .flags = flags);
      else
         nir_load_scs_base_agx(b, t->words, 32, t->base, .base = t->offset,
                               .format = PIPE_FORMAT_R32_UINT, .component = 0,
                               .flags = flags);
   }
}

bool
agx_nir_lower_gen1_ray_query(nir_shader *shader,
                           const struct agx_gen1_ray_query_layout *layout)
{
   const bool fragment = layout &&
      ((layout->fragment_shader && shader->info.stage == MESA_SHADER_FRAGMENT) ||
       (layout->graphics_stage && (shader->info.stage == MESA_SHADER_VERTEX ||
                                   shader->info.stage == MESA_SHADER_TESS_EVAL)));
   const bool kernel = layout && layout->compute_kernel && !fragment &&
      (shader->info.stage == MESA_SHADER_VERTEX ||
       shader->info.stage == MESA_SHADER_TESS_CTRL ||
       shader->info.stage == MESA_SHADER_TESS_EVAL ||
       shader->info.stage == MESA_SHADER_GEOMETRY);
   if (!layout || (!fragment && !kernel && shader->info.stage != MESA_SHADER_COMPUTE) ||
       (!fragment && !kernel && (shader->info.workgroup_size_variable ||
        !shader->info.workgroup_size[0] || !shader->info.workgroup_size[1] ||
        !shader->info.workgroup_size[2])) || layout->payload_uniform > 508 ||
       (layout->payload_uniform & 1) || layout->bvh_uniform > 508 ||
       (layout->bvh_uniform & 1) || layout->status_uniform > 510 ||
       (layout->status_uniform & 1) || layout->context_uniform > 508 ||
       (layout->context_uniform & 1) || layout->as_table_uniform > 508 ||
       (layout->as_table_uniform & 1) ||
       (!!layout->context_uniform != !!layout->as_table_uniform))
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   gen1_rq_alias_copies(impl);
   struct util_dynarray queries = UTIL_DYNARRAY_INIT;
   struct util_dynarray ops = UTIL_DYNARRAY_INIT;
   unsigned refusal_line = 0;
#define GEN1_RQ_UNSUPPORTED() do { refusal_line = __LINE__; goto unsupported; } while (0)
   nir_foreach_function(f, shader) {
      if (!f->impl)
         continue;
      nir_foreach_block(block, f->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!nir_intrinsic_is_ray_query(intr->intrinsic))
               continue;
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            if (f->impl != impl || !deref)
               GEN1_RQ_UNSUPPORTED();
            nir_deref_instr *root = deref;
            while (root->deref_type == nir_deref_type_array)
               root = nir_deref_instr_parent(root);
            if (root->deref_type != nir_deref_type_var ||
                !root->var->data.ray_query)
               GEN1_RQ_UNSUPPORTED();
            bool found = false;
            util_dynarray_foreach(&queries, struct gen1_rq_state, state)
               found |= state->query == root->var;
            if (!found) {
               struct gen1_rq_state state = {.query = root->var};
               util_dynarray_append(&queries, state);
            }
            switch (intr->intrinsic) {
            case nir_intrinsic_rq_initialize:
               /* Only the captured force-opaque path is mapped so far.
                * Do not guess mappings for other Vulkan ray flags. */
               if ((!layout->candidates &&
                    (!nir_src_is_const(intr->src[2]) ||
                     nir_src_as_uint(intr->src[2]) != 1)) ||
                   intr->src[1].ssa->bit_size != 64 ||
                   (!layout->instance_traversal &&
                    (!nir_src_is_const(intr->src[3]) ||
                     nir_src_as_uint(intr->src[3]) != 255)))
                  GEN1_RQ_UNSUPPORTED();
               break;
            case nir_intrinsic_rq_load:
               if (!gen1_rq_supported_load(intr, layout->instance_traversal, layout->position_fetch) &&
                   !(layout->candidates && gen1_candidate_supported_load(intr, layout->position_fetch)))
                  GEN1_RQ_UNSUPPORTED();
               break;
            case nir_intrinsic_rq_generate_intersection:
            case nir_intrinsic_rq_confirm_intersection:
               if (!layout->candidates) GEN1_RQ_UNSUPPORTED();
               break;
            case nir_intrinsic_rq_proceed:
            case nir_intrinsic_rq_terminate:
               break;
            default:
               GEN1_RQ_UNSUPPORTED();
            }
            util_dynarray_append(&ops, intr);
         }
      }
   }
   if (!util_dynarray_num_elements(&queries, struct gen1_rq_state))
      GEN1_RQ_UNSUPPORTED();

   util_dynarray_foreach(&queries, struct gen1_rq_state, state) {
      state->flags = nir_local_variable_create(impl,
         gen1_rq_field_type(state->query->type, glsl_uint_type()), "rq_flags");
      state->as =
         nir_local_variable_create(impl,
            gen1_rq_field_type(state->query->type, glsl_uint64_t_type()), "rq_as");
      state->origin = nir_local_variable_create(
         impl, gen1_rq_field_type(state->query->type,
            glsl_vector_type(GLSL_TYPE_FLOAT, 3)), "rq_origin");
      state->direction = nir_local_variable_create(
         impl, gen1_rq_field_type(state->query->type,
            glsl_vector_type(GLSL_TYPE_FLOAT, 3)), "rq_direction");
      state->tmin =
         nir_local_variable_create(impl,
            gen1_rq_field_type(state->query->type, glsl_float_type()), "rq_tmin");
      state->tmax =
         nir_local_variable_create(impl,
            gen1_rq_field_type(state->query->type, glsl_float_type()), "rq_tmax");
      state->mask =
         nir_local_variable_create(impl,
            gen1_rq_field_type(state->query->type, glsl_uint_type()), "rq_mask");
      state->active =
         nir_local_variable_create(impl,
            gen1_rq_field_type(state->query->type, glsl_bool_type()), "rq_active");
      state->started = nir_local_variable_create(impl,
         gen1_rq_field_type(state->query->type, glsl_bool_type()), "rq_started");
      state->cursor = nir_local_variable_create(impl,
         gen1_rq_field_type(state->query->type, glsl_uint_type()), "rq_cursor");
      state->refills = nir_local_variable_create(impl,
         gen1_rq_field_type(state->query->type, glsl_uint_type()), "rq_refills");
      for (unsigned i = 0; i < 4; ++i)
         state->candidate[i] = nir_local_variable_create(impl,
            gen1_rq_field_type(state->query->type, glsl_uvec4_type()), "rq_candidate");
      for (unsigned i = 0; i < 4; ++i)
         state->result[i] =
            nir_local_variable_create(impl,
               gen1_rq_field_type(state->query->type, glsl_uvec4_type()), "rq_packet");
   }

   unsigned hw_fields = 0;
   util_dynarray_foreach(&ops, nir_intrinsic_instr *, op)
      hw_fields |= gen1_rq_hw_fields(*op);
   bool launches = false;

   nir_builder b = nir_builder_create(impl);
   /* A compute kernel may launch several queries in a thread, or none, so
    * each of its threads keeps track of whether it has launched one.
    *
    * What a launch needs that is the same for the whole dispatch is stored
    * ahead of a thread's first launch only: the query context and the
    * constant words of the warp's save area. A later query of the thread
    * finds them in place and stores just its own ray and query words, ten
    * SCS stores fewer than the first. ASAHI_G17P_RQ_CTX_ONCE=0 stores them
    * ahead of every launch.
    *
    * A thread that launched nothing ends without the sweep documented above
    * gen1_rq_scs_sweep(): it used none of the storage the sweep names.
    * ASAHI_G17P_RQ_SCS_SWEEP_SKIP=0 has every thread issue it. */
   const bool compute_candidates = layout->candidates &&
      shader->info.stage == MESA_SHADER_COMPUTE;
   const unsigned sweep = compute_candidates
      ? debug_get_num_option("ASAHI_G17P_RQ_SCS_SWEEP", GEN1_RQ_SWEEP_DEFAULT)
      : 0;
   const bool skip_sweep = sweep &&
      debug_get_bool_option("ASAHI_G17P_RQ_SCS_SWEEP_SKIP", true);
   const bool context_once = compute_candidates &&
      debug_get_bool_option("ASAHI_G17P_RQ_CTX_ONCE", true);
   nir_variable *launched = NULL;
   if (skip_sweep || context_once) {
      launched = nir_local_variable_create(impl, glsl_bool_type(),
                                           "rq_launched");
      b.cursor = nir_before_impl(impl);
      nir_store_var(&b, launched, nir_imm_false(&b), 1);
   }
   util_dynarray_foreach(&ops, nir_intrinsic_instr *, op) {
      nir_intrinsic_instr *intr = *op;
      b.cursor = nir_before_instr(&intr->instr);
      nir_deref_instr *query = nir_src_as_deref(intr->src[0]);
      struct gen1_rq_state *state = NULL;
      util_dynarray_foreach(&queries, struct gen1_rq_state, candidate) {
         if (candidate->query == nir_deref_instr_get_variable(query)) {
            state = candidate;
            break;
         }
      }
      assert(state);
      nir_deref_instr *flags = gen1_rq_field_deref(&b, query, state->flags);
      nir_deref_instr *as = gen1_rq_field_deref(&b, query, state->as);
      nir_deref_instr *origin = gen1_rq_field_deref(&b, query, state->origin);
      nir_deref_instr *direction = gen1_rq_field_deref(&b, query, state->direction);
      nir_deref_instr *tmin = gen1_rq_field_deref(&b, query, state->tmin);
      nir_deref_instr *tmax = gen1_rq_field_deref(&b, query, state->tmax);
      nir_deref_instr *mask = gen1_rq_field_deref(&b, query, state->mask);
      nir_deref_instr *active = gen1_rq_field_deref(&b, query, state->active);
      nir_deref_instr *result[4];
      for (unsigned i = 0; i < 4; ++i)
         result[i] = gen1_rq_field_deref(&b, query, state->result[i]);
      nir_deref_instr *started=gen1_rq_field_deref(&b,query,state->started);
      nir_deref_instr *cursor=gen1_rq_field_deref(&b,query,state->cursor);
      nir_deref_instr *refills=gen1_rq_field_deref(&b,query,state->refills);
      nir_deref_instr *candidate[4];
      for(unsigned i=0;i<4;i++) candidate[i]=gen1_rq_field_deref(&b,query,state->candidate[i]);
      nir_def *value = NULL;
      switch (intr->intrinsic) {
      case nir_intrinsic_rq_initialize:
         if (b.shader->info.stage == MESA_SHADER_COMPUTE) {
            const char *capture = getenv("ASAHI_G17P_CAPTURE_SCS_INDEX");
            if (capture && *capture) {
               char *end;
               uint64_t target = strtoull(capture, &end, 0);
               if (!*end && target && !(target & 15)) {
                  nir_def *gid = nir_channel(&b,
                     nir_load_global_invocation_id(&b, 32), 0);
                  nir_def *physical = nir_load_scs_thread_index_agx(&b);
                  nir_store_global(&b, nir_vec4(&b, gid, physical,
                     nir_imm_int(&b, 0x53435349), nir_imm_int(&b, 0)),
                     nir_iadd(&b, nir_imm_int64(&b, target),
                        nir_u2u64(&b, nir_ishl_imm(&b, gid, 4))),
                     .align_mul = 16, .write_mask = 15);
               }
            }
         }
         nir_store_deref(&b,started,nir_imm_false(&b),1);
         nir_store_deref(&b,cursor,nir_imm_int(&b,0),1);
         nir_store_deref(&b,refills,nir_imm_int(&b,0),1);
         nir_store_deref(&b, flags, intr->src[2].ssa, 1);
         nir_store_deref(&b, as, intr->src[1].ssa, 1);
         nir_store_deref(&b, origin, intr->src[4].ssa, 7);
         nir_store_deref(&b, direction, intr->src[6].ssa, 7);
         nir_store_deref(&b, tmin, intr->src[5].ssa, 1);
         nir_store_deref(&b, tmax, intr->src[7].ssa, 1);
         nir_store_deref(&b, mask, intr->src[3].ssa, 1);
         nir_store_deref(&b, active, nir_ine_imm(&b, intr->src[1].ssa, 0), 1);
         for (unsigned i = 0; i < 4; ++i)
            nir_store_deref(&b, result[i], nir_imm_ivec4(&b, 0, 0, 0, 0), 15);
         /* Committed distance is initialized to tmax before traversal. */
         nir_store_deref(&b, result[1],
                       nir_vec4(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                                intr->src[7].ssa, nir_imm_int(&b, 0)),
                       15);
         break;
      case nir_intrinsic_rq_proceed: {
         launches = true;
         nir_variable *more=nir_local_variable_create(impl,glsl_bool_type(),"rq_more");
         nir_store_var(&b,more,nir_imm_false(&b),1);
         nir_push_if(&b, nir_load_deref(&b, active));
         if(layout->candidates) nir_push_if(&b,nir_inot(&b,nir_load_deref(&b,started)));
         nir_def *address = nir_load_deref(&b, as);
         nir_def *selector = nir_load_preamble(&b, 1, 64, .base = layout->bvh_uniform);
         if (layout->context_uniform) {
            if (context_once)
               nir_push_if(&b, nir_inot(&b, nir_load_var(&b, launched)));
            nir_def *context=nir_load_preamble(&b,1,64,.base=layout->context_uniform);
            for(unsigned i=0;i<6;i++) {
               nir_def *header = nir_load_global(&b,4,32,
                  nir_iadd_imm(&b,context,16*i),.align_mul=16);
               if (i == 3 && !fragment && layout->dispatch_group_count_offset) {
                  nir_def *groups_ptr = nir_load_global_constant(&b, 1, 64,
                     nir_iadd_imm(&b, nir_load_root_agx(&b),
                        layout->dispatch_group_count_offset),
                     .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
                  nir_def *groups = nir_load_global_constant(&b, 3, 32,
                     groups_ptr, .align_mul = 4,
                     .access = ACCESS_CAN_SPECULATE);
                  nir_def *global = nir_imul(&b,
                     groups, nir_imm_ivec3(&b,
                        b.shader->info.workgroup_size[0],
                        b.shader->info.workgroup_size[1],
                        b.shader->info.workgroup_size[2]));
                  header = nir_vec4(&b, nir_channel(&b, header, 0),
                     nir_channel(&b, header, 1), nir_channel(&b, global, 1),
                     nir_channel(&b, global, 2));
               }
               nir_store_scs_agx(&b,header,.base=0x6000+16*i);
            }
            nir_store_scs_agx(&b,nir_load_global(&b,fragment ? 4 : 2,32,
               nir_iadd_imm(&b,context,104),.align_mul=8),.base=0x6068);
            if (context_once)
               nir_pop_if(&b, NULL);
         }
         const bool stable_as_selector = layout->instance_traversal &&
            layout->as_table_uniform;
         if (stable_as_selector) {
            selector = nir_u2u64(&b, nir_load_global(&b, 1, 32,
               nir_iadd_imm(&b, address, 176), .align_mul = 4));
         } else if (layout->as_table_uniform) {
            nir_def *entry = nir_iadd_imm(&b, nir_load_scs_thread_index_agx(&b),
               layout->as_table_first ? layout->as_table_first : 1);
            nir_def *table = nir_load_preamble(&b, 1, 64,
                                                .base = layout->as_table_uniform);
            nir_store_global(&b, address, nir_iadd(&b, table,
               nir_u2u64(&b, nir_ishl_imm(&b, entry, 4))),
               .align_mul = 16, .write_mask = 1);
            nir_barrier(&b, .execution_scope = SCOPE_INVOCATION,
               .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_RELEASE,
               .memory_modes = nir_var_mem_global);
            selector = nir_u2u64(&b, entry);
         }
         nir_def *lo = nir_load_global(
            &b, 3, 32, nir_iadd_imm(&b, address, 0x50), .align_mul = 16);
         nir_def *hi = nir_load_global(
            &b, 3, 32, nir_iadd_imm(&b, address, 0x5c), .align_mul = 4);
         nir_def *valid_bounds = nir_iand(&b, nir_ball_fequal3(&b, lo, lo),
                                               nir_ball_fequal3(&b, hi, hi));
         if (stable_as_selector)
            valid_bounds = nir_iand(&b, valid_bounds,
                                    nir_ine_imm(&b, selector, 0));
         nir_push_if(&b, valid_bounds);
         nir_def *first_launch = NULL;
         if (launched) {
            if (context_once)
               first_launch = nir_inot(&b, nir_load_var(&b, launched));
            nir_store_var(&b, launched, nir_imm_true(&b), 1);
         }
         nir_def *base=nir_load_preamble(&b,1,64,.base=layout->payload_uniform);
         nir_def *id=nir_load_scs_thread_index_agx(&b);
         nir_def *index=nir_ishl_imm(&b,id,5);
         nir_def *lane=nir_iand_imm(&b,id,31);
         nir_def *query=nir_iadd_imm(&b,
            nir_imul_imm(&b,nir_ushr_imm(&b,id,5),layout->candidates ? 304*32 : layout->instance_traversal ? 0xb00 : 0x500),0x2f8000);
         nir_def *savebase=nir_load_preamble(&b,1,64,.base=layout->candidates ? 116 : 122);
         nir_def *saveindex=nir_imul_imm(&b,nir_ushr_imm(&b,id,5),12);
         /* The words of the save area that no launch changes. The two a
          * launch publishes are stored below, ahead of every launch. */
         if (first_launch)
            nir_push_if(&b, first_launch);
         nir_store_scs_uniform_agx(&b, nir_imm_ivec4(&b, 0, 0, 0, 0), savebase, saveindex,
            .base=256,.format=PIPE_FORMAT_R32_UINT,.component=2);
         nir_store_scs_uniform_agx(&b,nir_vec4(&b,
            nir_load_preamble(&b,1,32,.base=36),nir_load_preamble(&b,1,32,.base=38),
            nir_load_preamble(&b,1,32,.base=0),nir_load_preamble(&b,1,32,.base=2)),
            savebase,saveindex,.base=272,.format=PIPE_FORMAT_R32_UINT,.component=2);
         const unsigned ift_reference_uniform = 80;
         nir_store_scs_uniform_agx(&b,nir_vec2(&b,
            nir_load_preamble(&b,1,32,.base=ift_reference_uniform),
            nir_load_preamble(&b,1,32,.base=ift_reference_uniform + 2)),
            savebase,saveindex,.base=288,.format=PIPE_FORMAT_R32_UINT,.component=2);
         if (first_launch)
            nir_pop_if(&b, NULL);
         nir_def *words[2];
         agx_nir_gen1_ray_core_words(&b,nir_load_deref(&b,origin),
            nir_load_deref(&b,direction),nir_load_deref(&b,tmax),lo,hi,
            nir_imm_int(&b,0),nir_imm_false(&b),words);
         for(unsigned i=0;i<2;i++)
            nir_store_scs_uniform_agx(&b,words[i],base,index,
               .base=16*i,.format=PIPE_FORMAT_R32_UINT,.component=0);
         nir_def *query_control = layout->candidates
            ? gen1_rq_control(&b, flags)
            : nir_imm_int(&b, 0x4922);
         nir_store_scs_indexed_agx(&b,nir_vec2(&b,query_control,
            nir_load_deref(&b,tmin)),query,lane,.base=0,.component=3);
         nir_store_scs_indexed_agx(&b,nir_imm_ivec2(&b,0,0),query,lane,
            .base=256,.component=3);
         if (layout->instance_traversal)
            nir_store_scs_indexed_agx(&b,nir_vec2(&b,
               nir_iand_imm(&b,nir_load_deref(&b,mask),255),nir_imm_int(&b,0)),
               query,lane,.base=layout->candidates ? 8320 : 1792,.component=3);
         /* Native Gen1 publishes {0,1} to the saved warp context before launch. */
         nir_store_scs_uniform_agx(&b,nir_imm_ivec2(&b,0,1),savebase,saveindex,
            .base=256,.format=PIPE_FORMAT_R32_UINT,.component=2);
         if(layout->candidates)
            nir_store_scs_indexed_agx(&b,nir_imm_int(&b,8),query,lane,.base=1792,.component=2);
         nir_def *completed=agx_nir_gen1_trace_opaque_with_status(&b,selector,
            nir_load_preamble(&b,1,64,.base=layout->candidates ? 12 : 112),query,lane,
            nir_iadd_imm(&b,id,layout->launch_header),layout->candidates ? 123 : 126,layout->status_uniform);
         if(layout->candidates) {
            nir_store_deref(&b,started,nir_imm_true(&b),1);
            nir_store_deref(&b,active,completed,1);
            gen1_rq_merge_hw_commit(&b,id,query,lane,flags,active,result,hw_fields);
         } else {
         nir_push_if(&b,completed);
         for(unsigned i=0;i<2;i++)
            nir_store_deref(&b,result[i],nir_load_scs_uniform_agx(&b,4,32,
               base,index,.base=16*i,.format=PIPE_FORMAT_R32_UINT,.component=0),15);
         for(unsigned i=2;i<4;i++) {
            nir_def *v=nir_load_scs_indexed_agx(&b,2,32,query,lane,
               .base=256*(i-1),.component=3);
            nir_store_deref(&b,result[i],nir_vec4(&b,nir_channel(&b,v,0),
               nir_channel(&b,v,1),nir_imm_int(&b,0),nir_imm_int(&b,0)),15);
         }
         if (layout->instance_traversal) {
            nir_def *packet=nir_load_deref(&b,result[2]);
            nir_def *hit=nir_ine_imm(&b,nir_ushr_imm(&b,nir_channel(&b,packet,0),30),0);
            nir_def *zero=nir_imm_ivec2(&b,0,0);
            nir_push_if(&b,hit);
            /* Native G17P packs each 48-bit instance pointer into three
             * consecutive halfwords, at a six-byte per-lane stride. */
            nir_def *at=nir_imul_imm(&b,lane,6);
            nir_def *halves[3];
            for (unsigned i=0;i<3;i++)
               halves[i]=nir_u2u32(&b,nir_load_scs_indexed_agx(&b,1,16,
                  at,query,.base=2432+2*i,.component=0));
            nir_def *pointer=nir_pack_64_2x32_split(&b,
               nir_ior(&b,halves[0],nir_ishl_imm(&b,halves[1],16)),halves[2]);
            nir_def *ids=nir_load_global(&b,2,32,nir_iadd_imm(&b,pointer,96),.align_mul=4);
            nir_pop_if(&b,NULL);
            ids=nir_if_phi(&b,ids,zero);
            nir_store_deref(&b,result[2],nir_vec4(&b,
               nir_channel(&b,packet,0),nir_channel(&b,packet,1),
               nir_channel(&b,ids,1),nir_channel(&b,ids,0)),15);
         }
         nir_pop_if(&b,NULL);
         nir_store_deref(&b, active, nir_imm_false(&b), 1);
         }
         nir_push_else(&b, NULL);
         nir_store_deref(&b, active, nir_imm_false(&b), 1);
         nir_pop_if(&b, NULL);
         if (layout->candidates) {
            nir_pop_if(&b, NULL);
            nir_push_if(&b, nir_load_deref(&b, active));
            gen1_candidate_next(&b,active,cursor,flags,tmin,result,candidate,refills,more,hw_fields);
            nir_pop_if(&b, NULL);
         }
         nir_pop_if(&b, NULL);
         /* Opaque traversal consumes the whole query, with no candidate for
          * application code to confirm. Subsequent proceed calls are inert. */
         value = nir_load_var(&b,more);
         break;
      }
      case nir_intrinsic_rq_generate_intersection: {
         nir_def *distance=intr->src[1].ssa;
         nir_def *best=nir_channel(&b,nir_load_deref(&b,result[1]),2);
         nir_push_if(&b,nir_iand(&b,nir_fge(&b,distance,nir_load_deref(&b,tmin)),nir_fge(&b,best,distance)));
         for(unsigned i=0;i<4;i++)nir_store_deref(&b,result[i],nir_load_deref(&b,candidate[i]),15);
         nir_def *packet=nir_load_deref(&b,result[1]);
         nir_store_deref(&b,result[1],nir_vec4(&b,nir_channel(&b,packet,0),nir_channel(&b,packet,1),distance,nir_channel(&b,packet,3)),15);
         nir_push_if(&b,nir_test_mask(&b,nir_load_deref(&b,flags),4));
         nir_store_deref(&b,active,nir_imm_false(&b),1);
         nir_pop_if(&b,NULL);nir_pop_if(&b,NULL);
         break;
      }
      case nir_intrinsic_rq_confirm_intersection:
         for(unsigned i=0;i<4;i++)nir_store_deref(&b,result[i],nir_load_deref(&b,candidate[i]),15);
         nir_push_if(&b,nir_test_mask(&b,nir_load_deref(&b,flags),4));
         nir_store_deref(&b,active,nir_imm_false(&b),1);
         nir_pop_if(&b,NULL);
         break;
      case nir_intrinsic_rq_terminate:
         nir_store_deref(&b, active, nir_imm_false(&b), 1);
         break;
      case nir_intrinsic_rq_load:
         if(layout->candidates && !nir_intrinsic_committed(intr))
            for(unsigned i=0;i<4;i++)result[i]=candidate[i];
         switch (nir_intrinsic_ray_query_value(intr)) {
         case nir_ray_query_value_flags:
            value = nir_load_deref(&b, flags);
            break;
         case nir_ray_query_value_tmin:
            value = nir_load_deref(&b, tmin);
            break;
         case nir_ray_query_value_world_ray_origin:
            value = nir_load_deref(&b, origin);
            break;
         case nir_ray_query_value_world_ray_direction:
            value = nir_load_deref(&b, direction);
            break;
         case nir_ray_query_value_intersection_type:
            value = layout->candidates && !nir_intrinsic_committed(intr) ?
               nir_iadd_imm(&b,nir_ushr_imm(&b,nir_channel(&b,nir_load_deref(&b,result[2]),0),30),-1) : nir_ushr_imm(
                  &b,nir_channel(&b,nir_load_deref(&b,result[2]),0),30);
            break;
         case nir_ray_query_value_intersection_candidate_aabb_opaque:
            value=nir_ine_imm(&b,nir_channel(&b,nir_load_deref(&b,result[0]),2),0);
            break;
         case nir_ray_query_value_intersection_t:
            value = nir_channel(&b, nir_load_deref(&b, result[1]), 2);
            break;
         case nir_ray_query_value_intersection_primitive_index:
            value = nir_iand_imm(
               &b, nir_channel(&b, nir_load_deref(&b, result[2]), 0), 0x3fffffff);
            break;
         case nir_ray_query_value_intersection_geometry_index:
            value = nir_iand_imm(&b,
               nir_channel(&b, nir_load_deref(&b, result[2]), 1), 0xffffff);
            break;
         /* Gen1 agx.is_triangle_front_facing_intersection_result reads
          * the signed second committed-hit word from payload segment7. */
         case nir_ray_query_value_intersection_front_face:
            value = nir_ilt_imm(&b,
               nir_channel(&b, nir_load_deref(&b, result[2]), 1), 0);
            break;
         case nir_ray_query_value_intersection_triangle_vertex_positions: {
            nir_def *node = nir_pack_64_2x32(&b,
               nir_channels(&b, nir_load_deref(&b, result[0]), 3));
            nir_def *primitive = nir_iand_imm(
               &b, nir_channel(&b, nir_load_deref(&b, result[2]), 0), 0x3fffffff);
            nir_def *geometry = nir_iand_imm(&b,
               nir_channel(&b, nir_load_deref(&b, result[2]), 1), 0xffffff);
            value = nir_load_global(&b, 3, 32, nir_iadd_imm(&b,
               agx_nir_gen1_vertex_position_address(&b, node, geometry, primitive),
               12 * nir_intrinsic_column(intr)), .align_mul = 4);
            break;
         }
         case nir_ray_query_value_intersection_instance_sbt_index:
         case nir_ray_query_value_intersection_object_ray_direction:
         case nir_ray_query_value_intersection_object_ray_origin:
         case nir_ray_query_value_intersection_object_to_world:
         case nir_ray_query_value_intersection_world_to_object: {
            nir_def *node = nir_pack_64_2x32(&b,
               nir_channels(&b, nir_load_deref(&b, result[0]), 3));
            const nir_ray_query_value field = nir_intrinsic_ray_query_value(intr);
            if (field == nir_ray_query_value_intersection_instance_sbt_index ||
                field == nir_ray_query_value_intersection_object_to_world) {
               nir_def *relative = nir_load_global(&b, 1, 64,
                  nir_iadd_imm(&b, node, 104), .align_mul = 8);
               nir_def *transform = nir_iadd(&b, node, relative);
               if (field == nir_ray_query_value_intersection_instance_sbt_index) {
                  value = nir_iand_imm(&b, nir_load_global(&b, 1, 32,
                     nir_iadd_imm(&b, transform, 48), .align_mul = 4), 0xffffff);
               } else {
                  value = nir_load_global(&b, 3, 32,
                     nir_iadd_imm(&b, transform, 12 * nir_intrinsic_column(intr)),
                     .align_mul = 4);
               }
            } else if (field == nir_ray_query_value_intersection_world_to_object) {
               value = nir_load_global(&b, 3, 32,
                  nir_iadd_imm(&b, node, 16 + 12 * nir_intrinsic_column(intr)),
                  .align_mul = 4);
            } else {
               /* Native node stores four packed vec3 columns of world-to-
                * object at16. Translation applies to origins only. */
               const bool is_origin =
                  field == nir_ray_query_value_intersection_object_ray_origin;
               nir_def *ray = nir_load_deref(&b, is_origin ? origin : direction);
               nir_def *column[4];
               for (unsigned c = 0; c < 4; ++c)
                  column[c] = nir_load_global(&b, 3, 32,
                     nir_iadd_imm(&b, node, 16 + 12 * c), .align_mul = 4);
               nir_def *component[3];
               for (unsigned r = 0; r < 3; ++r) {
                  component[r] = is_origin ? nir_channel(&b, column[3], r)
                                           : nir_imm_float(&b, 0);
                  for (unsigned c = 0; c < 3; ++c)
                     component[r] = nir_ffma(&b, nir_channel(&b, ray, c),
                        nir_channel(&b, column[c], r), component[r]);
               }
               value = nir_vec3(&b, component[0], component[1], component[2]);
            }
            break;
         }
         case nir_ray_query_value_intersection_instance_id:
            value=nir_channel(&b,nir_load_deref(&b,result[2]),2);
            break;
         case nir_ray_query_value_intersection_instance_address:
            value = nir_pack_64_2x32(&b, nir_channels(&b, nir_load_deref(&b, result[0]), 3));
            break;
         case nir_ray_query_value_intersection_instance_custom_index:
            value=nir_channel(&b,nir_load_deref(&b,result[2]),3);
            break;
         case nir_ray_query_value_intersection_barycentrics:
            value = nir_channels(&b, nir_load_deref(&b, result[3]), 3);
            break;
         default:
            UNREACHABLE("preflighted Gen1 query load");
         }
         break;
      default:
         UNREACHABLE("preflighted Gen1 query operation");
      }
      if (value)
         nir_def_rewrite_uses(&intr->def, value);
      nir_instr_remove(&intr->instr);
   }
   /* Returns are lowered by now, so every thread reaches the end of the
    * entrypoint, whether or not it launched a ray. */
   if (sweep && launches) {
      b.cursor = nir_after_block_before_jump(nir_impl_last_block(impl));
      if (skip_sweep)
         nir_push_if(&b, nir_load_var(&b, launched));
      gen1_rq_scs_sweep(&b, layout, sweep);
      if (skip_sweep)
         nir_pop_if(&b, NULL);
   }
   util_dynarray_fini(&ops);
   util_dynarray_foreach(&queries, struct gen1_rq_state, state)
      state->query->data.ray_query = false;
   util_dynarray_fini(&queries);
   shader->info.ray_queries = 0;
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, shader, nir_lower_global_vars_to_local);
   NIR_PASS(_, shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, shader, nir_opt_dce);
   NIR_PASS(_, shader, nir_remove_dead_variables,
            nir_var_shader_temp | nir_var_function_temp, NULL);
   return true;
unsupported:
   if (debug_get_bool_option("ASAHI_G17P_RQ_LOWERING_DIAGNOSTIC", false))
      fprintf(stderr, "Gen1 ray-query refusal stage=%u line=%u queries=%zu operations=%zu instances=%d candidates=%d position_fetch=%d feature=%d\n",
              shader->info.stage, refusal_line,
              util_dynarray_num_elements(&queries, struct gen1_rq_state),
              util_dynarray_num_elements(&ops, nir_intrinsic_instr *), layout->instance_traversal, layout->candidates, layout->position_fetch, layout->diagnostic_features);
#undef GEN1_RQ_UNSUPPORTED
   util_dynarray_fini(&ops);
   util_dynarray_fini(&queries);
   return false;
}


nir_def *
agx_nir_gen1_candidate_count(nir_builder *b, nir_def *query, nir_def *lane)
{
   nir_def *word = nir_load_scs_indexed_agx(b, 1, 32, query, lane,
      .base=1792, .component=2);
   return nir_iand_imm(b, nir_ushr_imm(b, word, 21), 15);
}

struct agx_gen1_ray_candidate
agx_nir_gen1_load_candidate(nir_builder *b, nir_def *query,
                           nir_def *lane, nir_def *index)
{
   /* The native G17P compiler uses byte offsets in selector0, with the
    * query in slot A and lane/cursor offset in slot B. */
   nir_def *offset = nir_iadd(b, nir_imul_imm(b, lane, 200),
                                nir_imul_imm(b, index, 20));
   nir_def *prefix = nir_load_scs_indexed_agx(b, 2, 32, offset, query,
      .base=1920, .component=0);
   return (struct agx_gen1_ray_candidate){
      .compressed_address = nir_iand_imm(b, nir_pack_64_2x32(b, prefix),
                                         UINT64_C(0xffffffffffff)),
      .flags = nir_ushr_imm(b, nir_channel(b, prefix, 1), 16),
      .barycentrics = nir_load_scs_indexed_agx(b, 2, 32, offset, query,
         .base=1928, .component=0),
      .distance = nir_load_scs_indexed_agx(b, 1, 32, offset, query,
         .base=1936, .component=0),
   };
}

nir_def *
agx_nir_gen1_candidate_instance_pointer(nir_builder *b, nir_def *query,
                                        nir_def *lane)
{
   /* Native G17P indexes the three pointer halfwords with lane*3 in
    * selector1. The indexed intrinsic places its first source in slot B,
    * so the query precedes the lane offset here. */
   nir_def *at = nir_imul_imm(b, lane, 3), *half[3];
   for (unsigned i = 0; i < 3; ++i)
      half[i] = nir_u2u32(b, nir_load_scs_indexed_agx(b, 1, 16,
         query, at, .base = 9152 + 2 * i, .component = 1));
   return nir_pack_64_2x32_split(b,
      nir_ior(b, half[0], nir_ishl_imm(b, half[1], 16)), half[2]);
}

nir_def *
agx_nir_gen1_refill_candidate_queue(nir_builder *b, nir_def *query,
                                   nir_def *lane, nir_def *id,
                                   nir_def *best_distance)
{
   nir_def *control = nir_load_scs_indexed_agx(b, 1, 32, query, lane,
      .base = 1792, .component = 2);
   nir_store_scs_indexed_agx(b, nir_iand_imm(b, control, ~0x1e00001u),
      query, lane, .base = 1792, .component = 2);
   nir_store_scs_uniform_agx(b, best_distance,
      nir_load_preamble(b, 1, 64, .base = 118), nir_ishl_imm(b, id, 5),
      .base = 24, .format = PIPE_FORMAT_R32_UINT, .component = 0);
   nir_continue_ray_gen1_agx(b, id, .base = 1);
   return agx_nir_gen1_wait_traversal(b, query, lane, 128);
}
