APP              := div
$(APP)_BUILD_DIR := $(SN_ROOT)/sw/benchmarks/div/build
SRCS             := $(SN_ROOT)/sw/benchmarks/div/src/$(APP).c

include $(SN_ROOT)/sw/kernels/common.mk