APP              := idle
$(APP)_BUILD_DIR := $(SN_ROOT)/sw/benchmarks/idle/build
SRCS             := $(SN_ROOT)/sw/benchmarks/idle/src/$(APP).c

include $(SN_ROOT)/sw/kernels/common.mk