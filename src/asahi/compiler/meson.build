# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# SPDX-License-Identifier: MIT

libasahi_agx_files = files(
  'agx_compile.c',
  'agx_nir_lower_cmat.c',
  'agx_dce.c',
  'agx_liveness.c',
  'agx_insert_waits.c',
  'agx_nir_lower_address.c',
  'agx_nir_lower_denorm_preserve.c',
  'agx_nir_lower_fma_rte.c',
  'agx_nir_scalarize_int64.c',
  'agx_nir_lower_divmod64.c',
  'agx_nir_lower_atomics.c',
  'agx_nir_opt_flatten_loads.c',
  'agx_nir_opt_select_load.c',
  'agx_nir_opt_licm.c',
  'agx_nir_opt_sink_loop_sample_uses.c',
  'agx_nir_lower_clip_m1_1.c',
  'agx_nir_lower_cull_distance.c',
  'agx_nir_lower_fminmax.c',
  'agx_nir_lower_word_bounds.c',
  'agx_nir_share_bounds.c',
  'agx_nir_lower_frag_sidefx.c',
  'agx_nir_lower_sample_mask.c',
  'agx_nir_lower_discard_zs_emit.c',
  'agx_nir_lower_interpolation.c',
  'agx_nir_lower_g17p_combined_sampler.c',
  'agx_nir_ray_query.c',
  'agx_nir_gen1_ray_query.c',
  'agx_nir_lower_shared_bitsize.c',
  'agx_nir_lower_subgroups.c',
  'agx_nir_lower_compute_derivatives.c',
  'agx_nir_lower_workgroup_split.c',
  'agx_nir_lower_texture.c',
  'agx_nir_m5_mesh.c',
  'agx_nir_gen1_mesh.c',
  'agx3_mesh_core.c',
  'agx_nir_opt_preamble.c',
  'agx_lower_64bit.c',
  'agx_lower_divergent_shuffle.c',
  'agx_lower_parallel_copy.c',
  'agx_lower_pseudo.c',
  'agx_lower_spill.c',
  'agx_lower_uniform_sources.c',
  'agx_pack.c',
  'agx3_pack.c',
  'agx3_lower_uniform_program.c',
  'agx3_lower_cprog_inputs.c',
  'agx3_cprog_input.c',
  'agx3_select_core.c',
  'agx3_copy_core.c',
  'agx3_isel_core.c',
  'agx3_iadd_core.c',
  'agx3_pack_binary.c',
  'agx3_compose/agx3_compose.c',
  'agx3_compose/agx3_decode.c',
  'agx3_compose/agx3b_tables.c',
  'agx3_compose/agx3c_tables.c',
  'agx3_compose/agx3d_tables.c',
  'agx3_transcode.c',
  'agx3_binder.c',
  'agx3_atomic_core.c',
  'agx3_cf_core.c',
  'agx3_mem_core.c',
  'agx3_local_tps_core.c',
  'agx3_scs_core.c',
  'agx3_rt_m5_core.c',
  'agx3_subgroup_core.c',
  'agx3_matrix_core.c',
  'agx3_static_spill_core.c',
  'agx3_ioto_load_packer.c',
  'agx3_ioto_load_packer.h',
  'agx3_ioto_uniform_program.c',
  'agx3_ioto_uniform_program.h',
  'agx3_compute_program.c',
  'agx3_compute_program.h',
  'agx3_graphics_program.c',
  'agx3_m5_graphics_program.c',
  'agx3_graphics_program.h',
  'agx3_tex_core.c',
  'agx3_falu_mod_core.c',
  'agx3_uniform_alu.c',
  'agx3_uniform_integer.c',
  'agx3_vary_core.c',
  'agx3_tile_core.c',
  'agx3_pixel_core.c',
  'agx3_stage_core.c',
  'agx3_fltnorm_core.c',
  'agx3_writeout_core.c',
  'agx3_zs_core.c',
  'agx3_rt_core.c',
  'agx3_rt_gen1_entry.c',
  'agx3_rt_gen1_default.c',
  'agx3_rt_gen1_triangle.c',
  'agx3_rt_gen1_aabb.c',
  'agx3_rt_gen1_instance_push.c',
  'agx3_rt_gen1_runtime_constants.c',
  'agx3_rt_gen1_clique.c',
  'agx3_rt_gen1_runtime_tables.c',
  'agx3_rt_gen1_traversal.c',
  'agx3_slot_core.c',
  'agx3_resource_ir.c',
  'agx3_lower_texture_publications.c',
  'agx3_fixed_dependencies.c',
  'agx3_alu_core.c',
  'agx3_funop_core.c',
  'agx3_sr_core.c',
  'agx_performance.c',
  'agx_print.c',
  'agx_pressure_schedule.c',
  'agx_ir.c',
  'agx_opt_cse.c',
  'agx_opt_break_if.c',
  'agx_opt_empty_else.c',
  'agx_opt_jmp_none.c',
  'agx_opt_compact_constants.c',
  'agx_opt_promote_constants.c',
  'agx_opt_register_cache.c',
  'agx_optimizer.c',
  'agx_repair_ssa.c',
  'agx_reindex_ssa.c',
  'agx_spill.c',
  'agx_register_allocate.c',
  'agx_validate.c',
  'agx_validate_ra.c',
)

agx_nir_algebraic_c = custom_target(
  input : 'agx_nir_algebraic.py',
  output : 'agx_nir_algebraic.c',
  command : [
    prog_python, '@INPUT@', '-p', dir_compiler_nir,
  ],
  capture : true,
  depend_files : nir_algebraic_depends,
)

agx_opcodes_h = custom_target(
  input : ['agx_opcodes.h.py'],
  output : 'agx_opcodes.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('agx_opcodes.py'),
)

idep_agx_opcodes_h = declare_dependency(
  sources : [agx_opcodes_h],
  include_directories : include_directories('.'),
)

agx_opcodes_c = custom_target(
  input : ['agx_opcodes.c.py'],
  output : 'agx_opcodes.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('agx_opcodes.py'),
)

agx_builder_h = custom_target(
  input : ['agx_builder.h.py'],
  output : 'agx_builder.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('agx_opcodes.py'),
)

idep_agx_builder_h = declare_dependency(
  sources : [agx_builder_h],
  include_directories : include_directories('.'),
)

libasahi_compiler = static_library(
  'asahi_compiler',
  [libasahi_agx_files, agx_opcodes_c, agx_nir_algebraic_c],
  dependencies: [idep_nir, idep_agx_opcodes_h, idep_agx_builder_h, idep_mesautil, idep_libagx],
  link_with : [libagx2_disasm, libagx3c_disasm, libagx3d_disasm],
  c_args : [no_override_init_args, '-Wno-c2x-extensions',
            '-DAGX3_HAVE_AGX3D_TABLE'],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if with_tests
  # agx3_bind_compose()'s memo against the table walk, on every instruction
  # of the native witness texts.
  test(
    'agx3_bind_memo',
    executable(
      'agx3_bind_memo_test',
      files(
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3_decode.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_compose/agx3d_tables.c',
        'test/test-agx3-bind-memo.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args,
                '-DAGX3_HAVE_AGX3D_TABLE'],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      link_with : [libagx3c_disasm, libagx3d_disasm],
    ),
    suite : ['asahi'],
    args : [meson.project_source_root() / 'src/asahi/compiler/test/native/g18g'],
  )

  test(
    'agx3_subgroup_core',
    executable(
      'agx3_subgroup_core_test',
      files(
        'agx3_subgroup_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-subgroup-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_tex_core',
    executable(
      'agx3_tex_core_test',
      files('test/test-agx3-tex-core.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies : [idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_transcode',
    executable(
      'agx3_transcode_test',
      files('test/test-agx3-transcode.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies : [idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_falu_mod_core',
    executable(
      'agx3_falu_mod_core_test',
      files(
        'agx3_falu_mod_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-falu-mod-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_vary_core',
    executable(
      'agx3_vary_core_test',
      files(
        'agx3_vary_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-vary-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_cf_core',
    executable(
      'agx3_cf_core_test',
      files(
        'agx3_cf_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-cf-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_mem_core',
    executable(
      'agx3_mem_core_test',
      files(
        'agx3_mem_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-mem-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_alu_core',
    executable(
      'agx3_alu_core_test',
      files(
        'agx3_alu_core.c',
        'agx3_falu_mod_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'agx3_pack.c',
        'test/test-agx3-alu-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args, '-UNDEBUG'],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_g17p_producers',
    executable(
      'agx3_g17p_producers_test',
      files('test/test-agx3-g17p-producers.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies : [idep_nir, idep_agx_opcodes_h, idep_agx_builder_h,
                      idep_mesautil, idep_libagx],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_compose_corpus',
    executable(
      'agx3_compose_corpus_test',
      files(
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_compose/agx3d_tables.c',
        'test/test-agx3-compose-corpus.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_slot_core',
    executable(
      'agx3_slot_core_test',
      files(
        'agx3_slot_core.c',
        'test/test-agx3-slot-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_iadd_core',
    executable(
      'agx3_iadd_core_test',
      files(
        'agx3_iadd_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-iadd-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_isel_core',
    executable(
      'agx3_isel_core_test',
      files(
        'agx3_isel_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-isel-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_copy_core',
    executable(
      'agx3_copy_core_test',
      files(
        'agx3_copy_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-copy-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_ioto_load_packer',
    executable(
      'agx3_ioto_load_packer_test',
      files(
        'agx3_ioto_load_packer.c',
        'agx3_cprog_input.c',
        'agx3_static_spill_core.c',
        'agx3_mem_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        '../lib/agx_ioto_core.c',
        '../lib/agx_ioto_load_core.c',
        'test/test-agx3-ioto-load-packer.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_ioto_uniform_program',
    executable(
      'agx3_ioto_uniform_program_test',
      files(
        'agx3_copy_core.c',
        'agx3_ioto_uniform_program.c',
        'agx3_ioto_load_packer.c',
        'agx3_cprog_input.c',
        'agx3_static_spill_core.c',
        'agx3_mem_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        '../lib/agx_ioto_core.c',
        '../lib/agx_ioto_load_core.c',
        '../lib/agx_ioto_profile_core.c',
        '../lib/agx_ioto_state_core.c',
        '../lib/agx_ioto_preshader_core.c',
        '../lib/agx_ioto_program_core.c',
        'test/test-agx3-ioto-uniform-program.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      dependencies : [idep_mesautil],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_compile_fail_closed',
    executable(
      'agx3_compile_fail_closed_test',
      files('test/test-agx3-compile-fail-closed.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_cmat_mixed_mma8',
    executable(
      'agx3_cmat_mixed_mma8_test',
      files('test/test-agx3-cmat-mixed-mma8.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_wide_imad',
    executable(
      'agx3_wide_imad_test',
      files('test/test-agx3-wide-imad.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_int64',
    executable(
      'agx3_int64_test',
      files('test/test-agx3-int64.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_g17p_device_barrier',
    executable(
      'agx3_g17p_device_barrier_test',
      files('test/test-agx3-g17p-device-barrier.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_pixel_core',
    executable(
      'agx3_pixel_core_test',
      files(
        'agx3_pixel_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-pixel-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_funop_core',
    executable(
      'agx3_funop_core_test',
      files(
        'agx3_funop_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-funop-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_graphics_compile',
    executable(
      'agx3_graphics_compile_test',
      files('test/test-agx3-graphics-compile.c'),
      libagx[1],
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil, idep_libpoly],
      link_with : [libasahi_compiler, libpoly_nir],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_graphics_program',
    executable(
      'agx3_graphics_program_test',
      files(
        'agx3_graphics_program.c',
        'agx3_m5_graphics_program.c',
        'agx3_copy_core.c',
        'agx3_ioto_uniform_program.c',
        'agx3_ioto_load_packer.c',
        'agx3_cprog_input.c',
        'agx3_static_spill_core.c',
        'agx3_mem_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        '../lib/agx_ioto_core.c',
        '../lib/agx_ioto_load_core.c',
        '../lib/agx_ioto_profile_core.c',
        '../lib/agx_ioto_state_core.c',
        '../lib/agx_ioto_preshader_core.c',
        '../lib/agx_ioto_program_core.c',
        'test/test-agx3-graphics-program.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      dependencies : [idep_mesautil],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )
  test(
    'agx_tests',
    executable(
      'agx_tests',
      files(
         'test/test-compact-constants.cpp',
         'test/test-agx3-pack-binary.cpp',
         'test/test-minifloat.cpp',
         'test/test-optimizer.cpp',
         'test/test-pressure-cse.cpp',
         'test/test-rte-fma.cpp',
         'test/test-image-atomic-metadata.cpp',
         'test/test-ballot-count.cpp',
         'test/test-border-index.cpp',
         'test/test-aligned-bounds.cpp',
         'test/test-agx2-image-acquire.cpp',
         'test/test-byte-swap.cpp',
         'test/test-minmax-canonical.cpp',
         'test/test-workgroup-split.cpp',
         'test/test-selected-f2i.cpp',
         'test/test-buffer-format-groups.cpp',
         'test/test-scalarize-int64.cpp',
         'test/test-lower-divmod64.cpp',
         'test/test-spill-fallback.cpp',
         'test/test-agx-register-cache.cpp',
         'test/test-rt-register-pressure.cpp',
         'test/test-uniform-remat.cpp',
         'test/test-lower-pseudo.cpp',
         'test/test-lower-parallel-copy.cpp',
         'test/test-lower-spill.cpp',
         'test/test-repair-ssa.cpp',
         'test/test-preamble-image-payload.cpp',
         'test/test-preamble-share-values.cpp',
         'test/test-preamble-speculative-buffer.cpp',
         'test/test-preamble-modifiers.cpp',
         'test/test-preamble-texture-policy.cpp',
         'test/test-cprog-cleanup.cpp',
         'test/test-m5-packed-lod.cpp',
         'test/test-preamble-half-exports.cpp',
         'test/test-demote-early-kill.cpp',
         'test/test-preamble-half-word-exports.cpp',
         'test/test-quad-control.cpp',
         'test/test-preamble-priority.cpp',
         'test/test-signed-zero-fma.cpp',
         'test/test-negated-minifloat.cpp',
         'test/test-dual-absolute-select.cpp',
         'test/test-native-form-regressions.cpp',
         'test/test-inline-mrt.cpp',
         'test/test-mrt-terminal-tail.cpp',
         'test/test-agx3-cprog-spill-certificate.cpp',
         'test/test-agx3-sr.cpp',
         'test/test-agx3-scs.cpp',
         'test/test-agx3-register-copy.cpp',
         'test/test-agx3-uniform-alu.cpp',
         'test/test-position-order.cpp',
         'test/test-clip-m1-1.cpp',
         'test/test-mixed-uniform-alu.cpp',
         'test/test-agx3-integer-fusion.cpp',
         'test/test-agx3-integer-lowering.cpp',
         'test/test-agx3-unary-lowering.cpp',
         'test/test-agx3-fsel-uniform.cpp',
         'test/test-agx3-cube-scratch.cpp',
         'test/test-fminmax-constant-denorm.cpp',
         'test/test-denorm-preserve-fma.cpp',
         'test/test-fminmax-signed-zero.cpp',
         'test/test-shared-byte-offsets.cpp',
         'test/test-device-load-batches.cpp',
         'test/test-fuse-lea-constants.cpp',
         'test/test-divergence-load-agx.cpp',
         'test/test-lower-address-exact-scale.cpp',
         'test/test-lower-address-checked-offsets.cpp',
         'test/test-ftz-subgroup-minmax.cpp',
         'test/test-m5-half-widen-once.cpp',
         'test/test-m5-byte-to-float.cpp',
         'test/test-fold-narrowed-half.cpp',
         'test/test-m5-float-minmax-reduce.cpp',
         'test/test-m5-copy-schedule.cpp',
         'test/test-preamble-flag-branches.cpp',
         'test/test-flatten-loads.cpp',
         # agx_chip_isa_encoding() (test-native-form-regressions.cpp) lives
         # in libasahi_lib, which this test does not link.
         '../lib/agx_chip.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [idep_gtest, idep_nir, idep_agx_opcodes_h, idep_agx_builder_h, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )

  test(
    'agx3_rt_core',
    executable(
      'agx3_rt_core_test',
      files(
        'agx3_rt_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-rt-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_gen1_traversal',
    executable('agx3_gen1_traversal_test', files('test/test-gen1-traversal.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler]),
    suite : ['asahi'],
  )

  test(
    'agx3_rt_gen1_entry',
    executable(
      'agx3_rt_gen1_entry_test',
      files('test/test-agx3-rt-gen1-entry.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_scs_touch',
    executable(
      'agx3_scs_touch_test',
      files('test/test-agx3-scs-touch.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_gen1_scs_sweep',
    executable(
      'agx3_gen1_scs_sweep_test',
      files('test/test-gen1-scs-sweep.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test('agx3_gen1_wait',
    executable('agx3_gen1_wait_test', files('test/test-gen1-wait.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler]), suite : ['asahi'])
  test(
    'agx3_gen1_ray_query',
    executable('agx3_gen1_ray_query_test', files('test/test-gen1-ray-query.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler]),
    suite : ['asahi'],
  )
  test(
    'agx3_gen1_null_as',
    executable('agx3_gen1_null_as_test', files('test/test-gen1-null-as.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler]),
    suite : ['asahi'],
  )

  test(
    'agx3_gen1_terminate',
    executable('agx3_gen1_terminate_test', files('test/test-gen1-terminate.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler]),
    suite : ['asahi'],
  )

  test(
    'agx3_gen1_control',
    executable('agx3_gen1_control_test', files('test/test-gen1-control.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler]),
    suite : ['asahi'],
  )

  test(
    'agx3_gen1_scs_nir',
    executable(
      'agx3_gen1_scs_nir_test',
      files('test/test-gen1-scs-nir.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )
  test(
    'agx_nir_sink_loop_sample_uses',
    executable(
      'agx_nir_sink_loop_sample_uses_test',
      files('test/test-sink-loop-sample-uses.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_gen1_status',
    executable(
      'agx3_gen1_status_test',
      files('test/test-gen1-status.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_gen1_position_fetch',
    executable('agx3_gen1_position_fetch_test', files('test/test-gen1-position-fetch.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler]),
    suite : ['asahi'],
  )

  test(
    'agx3_ray_query_prerequisites',
    executable(
      'agx3_ray_query_prerequisites_test',
      files('test/test-agx3-ray-query-prerequisites.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_g17p_task_kernel_shared',
    executable(
      'agx3_g17p_task_kernel_shared_test',
      files('test/test-agx3-g17p-task-kernel-shared.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )
endif
