/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdint.h>
#include "agx_pack.h"
#include "shader_enums.h"
#include "../compiler/agx_mesh_abi.h"

struct nir_shader;

/* Matches the hardware order */

enum uvs_group {
   UVS_POSITION,
   UVS_VARYINGS,
   UVS_PSIZ,
   UVS_LAYER_VIEWPORT,
   UVS_CLIP_DIST,
   UVS_NUM_GROUP,
};

/**
 * Represents an "unlinked" UVS layout. This is computable from an unlinked
 * vertex shader without knowing the associated fragment shader. The various UVS
 * groups have fixed offsets, but the varyings within the varying group have
 * indeterminate order since we don't yet know the fragment shader interpolation
 * qualifiers.
 */
struct agx_unlinked_uvs_layout {
   /* Bit i set <===> components[i] != 0 && i != POS && i != PSIZ. For fast
    * iteration of user varyings.
    */
   uint64_t written;

   /* Fully packed data structure */
   struct agx_vdm_state_vertex_outputs_packed vdm;

   /* Partial data structure, must be merged with FS selects */
   struct agx_output_select_packed osel;

   /* Offset of each group in the UVS in words. */
   uint8_t group_offs[UVS_NUM_GROUP];

   /* Size of the UVS allocation in words. >= last group_offs element */
   uint8_t size;

   /* Size of the UVS_VARYINGS */
   uint8_t user_size;

   bool mesh_primitive_page : 1;

   /* The varyings are written twice in location order, a smooth copy and
    * then a flat copy of the same size, whatever the fragment shader's
    * interpolation qualifiers, so the shader writes them at constant UVS
    * indices: agx_nir_lower_uvs() with canonical, for a shader compiled
    * without its fragment shader. user_size counts both copies.
    */
   bool canonical : 1;

   /* The varyings are too many to be written twice, so the shader is
    * compiled without its fragment shader with each one at a placeholder
    * UVS index, the index agx_assign_uvs() gives it with no flat or linear
    * inputs, and lists those writes (agx_shader_info::agx3_uvs_reloc_count).
    * Linking it rewrites them for the fragment shader's qualifiers
    * (agx_uvs_relocation_map()), so no index is computed at run time.
    */
   bool relocatable : 1;

   /* Number of 32-bit components written for each slot. TODO: Model 16-bit.
    *
    * Invariant: sum_{slot} (components[slot]) =
    *            group_offs[PSIZ] - group_offs[VARYINGS]
    */
   uint8_t components[VARYING_SLOT_MAX];
};
static_assert(sizeof(struct agx_unlinked_uvs_layout) == 88, "packed");

/* Both copies of a canonical layout's varyings fit the G17P native vertex
 * program's 125 user words (HK_AGX3_G17P_MAX_USER_WORDS).
 */
#define AGX_UVS_CANONICAL_MAX_WORDS 124

/* Supply the API's default Layer for drivers transporting it as a varying. */
bool agx_nir_lower_default_layer(struct nir_shader *s);

bool agx_nir_lower_uvs(struct nir_shader *s,
                       struct agx_unlinked_uvs_layout *layout,
                       bool layer_as_varying, bool canonical);

/**
 * Represents a linked UVS layout.
 */
struct agx_varyings_vs {
   /* Associated linked hardware data structures */
   struct agx_varying_counts_packed counts_32, counts_16;

   /* If the user varying slot is written, this is the base index that the first
    * component of the slot is written to. The next components are found in the
    * next indices. Otherwise 0, aliasing position.
    */
   unsigned slots[VARYING_SLOT_MAX];

   /* For a canonical layout, the distance from a slot's smooth copy to its
    * flat copy, which flat CF bindings read. Otherwise 0.
    */
   unsigned flat_copy;
};

void agx_assign_uvs(struct agx_varyings_vs *varyings,
                    const struct agx_unlinked_uvs_layout *layout,
                    uint64_t flat_mask, uint64_t linear_mask);

/* For a relocatable layout, map each placeholder UVS index to the index the
 * layout linked with these fragment interpolation qualifiers gives the same
 * component (agx_assign_uvs()). Every other index maps to itself.
 */
void agx_uvs_relocation_map(const struct agx_unlinked_uvs_layout *layout,
                            uint64_t flat_mask, uint64_t linear_mask,
                            uint8_t map[256]);

struct agx_varyings_fs;

void agx_link_varyings_vs_fs(void *out, struct agx_varyings_vs *vs,
                             unsigned nr_user_indices,
                             struct agx_varyings_fs *fs,
                             unsigned provoking_vertex,
                             uint8_t sprite_coord_enable,
                             bool *generate_primitive_id);
