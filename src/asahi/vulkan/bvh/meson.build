# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

hk_bvh_include_dir = dir_source_root + '/src/asahi/vulkan/bvh'

hk_bvh_includes = files(
  'agx3_node.h',
  'hk_build_interface.h',
  '../hk_bvh.h',
)

# Reassigning bvh_spv follows the same pattern turnip uses: src/vulkan is
# subdir'd before src/asahi, so the common bvh_spv has already been consumed
# by then and vk_bvh_include_dir / vk_glsl_shader_preamble are in scope.
#
# The native node encoders are chosen by the device's chip
# (hk_agx3_as_format_for_chip()). GEN1 (G17P and the other native AGX3
# chips, T6040 included) builds from agx3_{leaf,envelope,plan,refit}.comp,
# the same sources its refit encoding uses; the hk_agx3_gen1_* programs are
# those sources. M5 (G17G/G17S) builds from agx3_m5_legacy_*.comp. Each AS
# format's records are written, refit, traversed and copied only by its own
# family.
hk_bvh_spv = []
foreach s : [['encode.comp', 'hk_encode'], ['write.comp', 'hk_bvh_write'],
             ['agx3_leaf.comp', 'hk_agx3_refit_encode', '-DHK_AGX3_REFIT=1'],
             ['agx3_leaf.comp', 'hk_agx3_refit_encode_triangles', '-DHK_AGX3_REFIT_TRIANGLES=1'],
             ['agx3_batch_build_leaf.comp', 'hk_agx3_batch_build_leaf'],
             ['agx3_refit.comp', 'hk_agx3_refit'],
             ['agx3_refit.comp', 'hk_agx3_refit_propagate', '-DHK_AGX3_REFIT_ONLY_PHASE=2u'],
             ['agx3_refit.comp', 'hk_agx3_refit_inner', '-DHK_AGX3_REFIT_ONLY_PHASE=5u'],
             ['agx3_refit_leaf.comp', 'hk_agx3_refit_leaf'],
             ['agx3_m5_legacy_leaf.comp', 'hk_agx3_m5_legacy_leaf'],
             ['agx3_m5_legacy_envelope.comp', 'hk_agx3_m5_legacy_envelope'],
             ['agx3_m5_legacy_plan.comp', 'hk_agx3_m5_legacy_plan'],
             ['agx3_m5_legacy_refit.comp', 'hk_agx3_m5_legacy_refit'],
             ['agx3_leaf.comp', 'hk_agx3_gen1_leaf'],
             ['agx3_leaf.comp', 'hk_agx3_gen1_leaf_pairs', '-DHK_AGX3_LEAF_PAIRS=1'],
             ['agx3_leaf.comp', 'hk_agx3_gen1_leaf_instances', '-DHK_AGX3_LEAF_ONLY=2'],
             ['agx3_envelope.comp', 'hk_agx3_gen1_envelope'],
             ['agx3_plan.comp', 'hk_agx3_gen1_plan'],
             ['agx3_plan.comp', 'hk_agx3_gen1_plan_pairs', '-DHK_AGX3_PLAN_PAIRS=1'],
             ['vertex_table.comp', 'hk_vertex_table']]
  command = [
    prog_glslang, '-V',
    '-I' + vk_bvh_include_dir,
    '-I' + hk_bvh_include_dir,
    '-I' + dir_source_root + '/src/asahi/vulkan',
    '--target-env', 'spirv1.5', '-x', '-o', '@OUTPUT@', '@INPUT@',
  ]
  command += s.length() > 2 ? [s[2]] : []
  command += glslang_quiet
  command += vk_glsl_shader_preamble

  hk_bvh_spv += custom_target(
    s[1] + '.spv.h',
    input : s[0],
    output : s[1] + '.spv.h',
    command : command,
    depend_files : [vk_bvh_includes, hk_bvh_includes],
  )
endforeach
