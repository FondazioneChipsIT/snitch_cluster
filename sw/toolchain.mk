# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>

###################
# Build variables #
###################

# Compiler toolchain
SN_LLVM_BINROOT  ?= $(dir $(shell which riscv32-unknown-elf-clang))
SN_RISCV_CC      ?= $(SN_LLVM_BINROOT)/clang
SN_RISCV_CXX     ?= $(SN_LLVM_BINROOT)/clang++
SN_RISCV_LD      ?= $(SN_LLVM_BINROOT)/ld.lld
SN_RISCV_AR      ?= $(SN_LLVM_BINROOT)/llvm-ar
SN_RISCV_OBJCOPY ?= $(SN_LLVM_BINROOT)/llvm-objcopy
SN_RISCV_OBJDUMP ?= $(SN_LLVM_BINROOT)/llvm-objdump

# Compiler flags
SN_RISCV_CFLAGS := -mcpu=snitch
SN_RISCV_CFLAGS += -menable-experimental-extensions
# SN_RISCV_CFLAGS += --target=riscv32-unknown-elf
# SN_RISCV_CFLAGS += -march=rv32imaf_xdma_xssr_xf16alt
SN_RISCV_CFLAGS += -mabi=ilp32d	
SN_RISCV_CFLAGS += -mcmodel=medany
SN_RISCV_CFLAGS += -mno-fdiv
SN_RISCV_CFLAGS += -fno-builtin-printf
SN_RISCV_CFLAGS += -fno-builtin-sqrtf
SN_RISCV_CFLAGS += -fno-common
SN_RISCV_CFLAGS += -fopenmp
SN_RISCV_CFLAGS += -ftls-model=local-exec
SN_RISCV_CFLAGS += -O3
SN_RISCV_CFLAGS += -Werror
SN_RISCV_CFLAGS += -Wno-unused-command-line-argument
SN_RISCV_CFLAGS += -ffast-math
# Use this flag to get warnings about implicit flota to double conversions
# These conversions can cause to exceptions and boot loops
# It is by default disabled because it also gives errors about fp16 to float conversions,
# which do not cause exceptions or issues.
# SN_RISCV_CFLAGS += -Wdouble-promotion

ifeq ($(DEBUG), ON)
SN_RISCV_CFLAGS += -g
endif
ifeq ($(OPENOCD_SEMIHOSTING), ON)
SN_RISCV_CFLAGS += -DOPENOCD_SEMIHOSTING
endif

# Linker flags
SN_RISCV_LDFLAGS := -fuse-ld=$(SN_RISCV_LD)
SN_RISCV_LDFLAGS += -nostartfiles
SN_RISCV_LDFLAGS += -nostdlib++
SN_RISCV_LDFLAGS += -lm

# Archiver flags
SN_RISCV_ARFLAGS := rcs

# Objdump flags
SN_RISCV_OBJDUMP_FLAGS := --mcpu=snitch
SN_RISCV_OBJDUMP_FLAGS += -D

# Header file in picolibc
# SN_RISCV_CFLAGS += -isystem /data/luca.colombo/llvm-snitch32/include
# SN_RISCV_LDFLAGS += -L/data/luca.colombo/llvm-snitch32/lib/
# SN_RISCV_LDFLAGS += -L/data/luca.colombo/llvm-snitch32/lib/clang/15/lib/linux