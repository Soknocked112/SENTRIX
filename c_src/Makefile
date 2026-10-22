# ==============================================================================
# Makefile for compiling eBPF XDP C Programs
# ==============================================================================

CC ?= clang
LLC ?= llc
ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/')
CLANG_FLAGS ?= -O2 -g -Wall -target bpf -D__TARGET_ARCH_$(ARCH)

SOURCES := xdp_ips.c
OBJECTS := $(SOURCES:.c=.o)

.PHONY: all clean check

all: $(OBJECTS)

%.o: %.c common_kern.h
	@which $(CC) > /dev/null 2>&1 || (echo "clang is required to compile eBPF bytecode. Install with: sudo apt install clang llvm"; exit 1)
	$(CC) $(CLANG_FLAGS) -I/usr/include -I/usr/include/$(shell uname -m)-linux-gnu -c $< -o $@

check:
	@echo "Checking eBPF source syntax..."
	$(CC) -fsyntax-only -Wall -Wextra -I. $(SOURCES) || true

clean:
	rm -f $(OBJECTS)
