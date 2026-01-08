# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

APP              := FFT
SRCS             := $(SN_ROOT)/sw/benchmarks/DSP/$(APP)/src/main.c
$(APP)_BUILD_DIR ?= $(SN_ROOT)/sw/benchmarks/DSP/$(APP)/build
$(APP)_INCDIRS   := $(SN_ROOT)/sw/benchmarks/DSP/$(APP)/data

include $(SN_ROOT)/sw/kernels/common.mk
