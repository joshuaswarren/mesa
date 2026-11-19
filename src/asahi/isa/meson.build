# Copyright 2025 Valve Corporation
# Copyright 2020-2021 Collabora
# SPDX-License-Identifier: MIT

# One decoder per instruction encoding: AGX2 for G13/G14, and the three AGX3
# tables -- agx3b for G15/G16/G17P, agx3c for G17, agx3d for G18G. Each
# generates its own decode tables under its own symbol prefix, taken from the
# XML basename.
agx_isa_xml = ['AGX2', 'AGX3b', 'AGX3c', 'AGX3d']
agx_disasm_libs = []

foreach isa : agx_isa_xml
  isa_lc = isa.to_lower()

  disasm_c = custom_target(
    input : ['gen-disasm.py', isa + '.xml'],
    output : isa_lc + '_disasm.c',
    command : [prog_python, '@INPUT@'],
    capture : true,
    depend_files : files('isa.py'),
  )

  agx_disasm_libs += static_library(
    isa_lc + '_disasm',
    disasm_c,
    include_directories : [inc_include, inc_src],
    c_args : [no_override_init_args],
    gnu_symbol_visibility : 'hidden',
    build_by_default : false,
  )
endforeach

libagx2_disasm = agx_disasm_libs[0]
libagx3b_disasm = agx_disasm_libs[1]
libagx3c_disasm = agx_disasm_libs[2]
libagx3d_disasm = agx_disasm_libs[3]

if with_tests
  test(
    'agx2_disasm',
    executable(
      'agx2_disasm_test',
      files('test/test-disassembler.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      link_with : [libagx2_disasm],
    ),
    suite : ['asahi'],
    args : files('test/cases.txt'),
  )

  # Every committed G17G/G18G native witness text, through the decoder for
  # its encoding. The G18G texts carry the 24-byte AGX3d smp form, so this is
  # what holds the decoder to its full instruction width.
  test(
    'agx3_native_corpus_disasm',
    executable(
      'agx3_native_corpus_disasm_test',
      files('test/test-agx3-native-corpus.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      link_with : [libagx3c_disasm, libagx3d_disasm],
    ),
    suite : ['asahi'],
    args : [meson.project_source_root() / 'src/asahi/compiler/test/native/g18g'],
  )

  # The indexed instruction length lookup against testing every form in
  # order, for every encoding, on random code and the native witness texts.
  test(
    'agx_instr_length_index',
    executable(
      'agx_instr_length_index_test',
      files('test/test-instr-length-index.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      link_with : agx_disasm_libs,
    ),
    suite : ['asahi'],
    args : [meson.project_source_root() / 'src/asahi/compiler/test/native/g18g'],
  )

 test(
    'agx2_python_disasm',
    prog_python,
    args : files('test-disasm.py', 'test/cases.txt'),
    suite : ['asahi']
  )
endif
