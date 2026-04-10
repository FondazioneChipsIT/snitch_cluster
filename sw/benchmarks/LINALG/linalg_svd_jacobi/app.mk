APP              := linalg_svd_jacobi
$(APP)_BUILD_DIR := $(SN_ROOT)/sw/benchmarks/LINALG/$(APP)/build
SRCS             := $(SN_ROOT)/sw/benchmarks/LINALG/$(APP)/src/$(APP).c
$(APP)_INCDIRS   := $(SN_ROOT)/sw/benchmarks/LINALG/$(APP)/data

include $(SN_ROOT)/sw/kernels/common.mk