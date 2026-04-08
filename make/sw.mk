# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

###################
# General targets #
###################

.PHONY: sn-sw sn-clean-sw

sn-sw: sn-runtime sn-tests sn-apps
sn-clean-sw: sn-clean-runtime sn-clean-tests sn-clean-apps

####################
# Platform headers #
####################

SN_RUNTIME_SRCDIR ?= $(SN_ROOT)/sw/runtime/impl

SN_SNITCH_CLUSTER_CFG_H                = $(SN_RUNTIME_SRCDIR)/snitch_cluster_cfg.h
SN_SNITCH_CLUSTER_ADDRMAP_H            = $(SN_RUNTIME_SRCDIR)/snitch_cluster_addrmap.h
SN_SNITCH_CLUSTER_RAW_ADDRMAP_H        = $(SN_RUNTIME_SRCDIR)/snitch_cluster_raw_addrmap.h
SN_SNITCH_CLUSTER_PERIPHERAL_H         = $(SN_RUNTIME_SRCDIR)/snitch_cluster_peripheral.h
SN_SNITCH_CLUSTER_PERIPHERAL_ADDRMAP_H = $(SN_RUNTIME_SRCDIR)/snitch_cluster_peripheral_addrmap.h
SN_SNITCH_CLUSTER_ADDRMAP_RDL          = $(SN_RUNTIME_SRCDIR)/snitch_cluster_addrmap.rdl

SN_SNITCH_CLUSTER_CFG_H_TPL       = $(SN_RUNTIME_SRCDIR)/snitch_cluster_cfg.h.tpl
SN_SNITCH_CLUSTER_ADDRMAP_RDL_TPL = $(SN_RUNTIME_SRCDIR)/snitch_cluster_addrmap.rdl.tpl

SN_RUNTIME_HAL_HDRS += $(SN_SNITCH_CLUSTER_CFG_H)
SN_RUNTIME_HAL_HDRS += $(SN_SNITCH_CLUSTER_ADDRMAP_H)
SN_RUNTIME_HAL_HDRS += $(SN_SNITCH_CLUSTER_RAW_ADDRMAP_H)
SN_RUNTIME_HAL_HDRS += $(SN_SNITCH_CLUSTER_PERIPHERAL_H)
SN_RUNTIME_HAL_HDRS += $(SN_SNITCH_CLUSTER_PERIPHERAL_ADDRMAP_H)

# CLUSTERGEN rules
$(eval $(call sn_cluster_gen_rule,$(SN_SNITCH_CLUSTER_CFG_H),$(SN_SNITCH_CLUSTER_CFG_H_TPL)))
$(eval $(call sn_cluster_gen_rule,$(SN_SNITCH_CLUSTER_ADDRMAP_RDL),$(SN_SNITCH_CLUSTER_ADDRMAP_RDL_TPL)))

# peakrdl headers
SN_PEAKRDL_INCDIRS += -I $(SN_ROOT)/hw/snitch_cluster/src/snitch_cluster_peripheral
SN_PEAKRDL_INCDIRS += -I $(SN_GEN_DIR)
$(eval $(call sn_peakrdl_generate_header_rule,$(SN_SNITCH_CLUSTER_PERIPHERAL_H),$(SN_ROOT)/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg.rdl))
$(eval $(call sn_peakrdl_generate_header_rule,$(SN_SNITCH_CLUSTER_ADDRMAP_H),$(SN_SNITCH_CLUSTER_ADDRMAP_RDL),$(SN_PEAKRDL_INCDIRS)))
# Addrmap depends on generated cluster RDL
$(SN_SNITCH_CLUSTER_ADDRMAP_H): $(SN_CLUSTER_RDL)

$(SN_SNITCH_CLUSTER_RAW_ADDRMAP_H): $(SN_SNITCH_CLUSTER_ADDRMAP_RDL) $(SN_CLUSTER_RDL)
	@echo "[peakrdl] Generating $@"
	$(SN_PEAKRDL) raw-header $< -o $(SN_SNITCH_CLUSTER_RAW_ADDRMAP_H) --base_name $(notdir $(basename $@)) --format c $(SN_PEAKRDL_INCDIRS)

$(SN_SNITCH_CLUSTER_PERIPHERAL_ADDRMAP_H): $(SN_ROOT)/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg.rdl
	@echo "[peakrdl] Generating $@"
	$(SN_PEAKRDL) raw-header $< -o $(SN_SNITCH_CLUSTER_PERIPHERAL_ADDRMAP_H) --format c $(SN_PEAKRDL_INCDIRS)

.PHONY: sn-clean-headers
sn-clean-sw: sn-clean-headers
sn-clean-headers:
	rm -f $(SN_RUNTIME_HAL_HDRS) $(SN_SNITCH_CLUSTER_ADDRMAP_RDL)

##################
# Subdirectories #
##################

include $(SN_ROOT)/sw/toolchain.mk
include $(SN_ROOT)/sw/runtime/runtime.mk
include $(SN_ROOT)/sw/tests/tests.mk
include $(SN_ROOT)/sw/riscv-tests/riscv-tests.mk

SN_BUILD_APPS ?= ON

ifeq ($(SN_BUILD_APPS), ON)
# Benhmarks
# DSP kernels
SN_APPS += $(SN_ROOT)/sw/benchmarks/DSP/CONV3x3
SN_APPS += $(SN_ROOT)/sw/benchmarks/DSP/DWT
SN_APPS += $(SN_ROOT)/sw/benchmarks/DSP/FFT
SN_APPS += $(SN_ROOT)/sw/benchmarks/DSP/FIR
SN_APPS += $(SN_ROOT)/sw/benchmarks/DSP/kmeans_b
# LLM kernels
SN_APPS += $(SN_ROOT)/sw/benchmarks/LLM/matmul_fp32
SN_APPS += $(SN_ROOT)/sw/benchmarks/LLM/residual
SN_APPS += $(SN_ROOT)/sw/benchmarks/LLM/attention_b
SN_APPS += $(SN_ROOT)/sw/benchmarks/LLM/gelu_b
SN_APPS += $(SN_ROOT)/sw/benchmarks/LLM/layernorm_b
SN_APPS += $(SN_ROOT)/sw/benchmarks/LLM/softmax_b

#LINALG kernels
SN_APPS += $(SN_ROOT)/sw/benchmarks/LINALG/linalg_cholesky_decomp
SN_APPS += $(SN_ROOT)/sw/benchmarks/LINALG/linalg_lu_decomp
SN_APPS += $(SN_ROOT)/sw/benchmarks/LINALG/linalg_lu_solve
SN_APPS += $(SN_ROOT)/sw/benchmarks/LINALG/gemm
SN_APPS += $(SN_ROOT)/sw/benchmarks/LINALG/gemv


endif

# Include Makefile from each app subdirectory
$(foreach app,$(SN_APPS), \
	$(eval include $(app)/app.mk) \
)
