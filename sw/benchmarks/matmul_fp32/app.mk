APP              := matmul_fp32
$(APP)_BUILD_DIR ?= $(SN_ROOT)/sw/benchmarks/$(APP)/build
SRC_DIR          := $(SN_ROOT)/sw/benchmarks/$(APP)/src
SRCS             := $(SRC_DIR)/$(APP).c

include $(SN_ROOT)/sw/kernels/common.mk
