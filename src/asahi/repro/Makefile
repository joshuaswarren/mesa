# Copyright 2026 Joshua Warren
# SPDX-License-Identifier: MIT
#
# Standalone build for the Honeykrisp miscompile reproductions. Not part
# of Mesa's meson build and not run by CI: these programs exist to be
# built and run by hand on the hardware under test.
#
#   make            build ./repro and the SPIR-V for every case
#   ./repro         run all cases; exit 0 when every check passes
#   ./repro 3       run only case 3 (and its control)
#
# glslangValidator comes from the glslang package on every major distro.

COMP  ?= glslangValidator
CC    ?= cc
CFLAGS ?= -O2 -Wall -Wextra

SHADERS = \
	01-word-load-divergent-loop \
	02-scan-dynamic-shift \
	03-selector-bytepath \
	03-noselector-bytepath \
	04-packed-bool-and \
	05-reduce-load-truthy

SPV = $(SHADERS:%=%.spv)

all: repro $(SPV)

repro: repro.c vk.c vk.h $(SPV)
	$(CC) $(CFLAGS) -o $@ repro.c vk.c -lvulkan

%.spv: %.comp
	$(COMP) -V $< -o $@

clean:
	rm -f repro $(SPV)

.PHONY: all clean
