APP              := CONV3
$(APP)_BUILD_DIR := $(SN_ROOT)/sw/benchmarks/DSP/$(APP)/build
SRCS             := $(SN_ROOT)/sw/benchmarks/DSP/$(APP)/src/$(APP).c
$(APP)_INCDIRS   := $(SN_ROOT)/sw/benchmarks/DSP/$(APP)/data

include $(SN_ROOT)/sw/kernels/common.mk