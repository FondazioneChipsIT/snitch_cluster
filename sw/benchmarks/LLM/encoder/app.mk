APP              := encoder
$(APP)_BUILD_DIR := $(SN_ROOT)/sw/benchmarks/LLM/$(APP)/build
SRCS             := $(SN_ROOT)/sw/benchmarks/LLM/$(APP)/src/$(APP).c
$(APP)_INCDIRS   := $(SN_ROOT)/sw/benchmarks/LLM/$(APP)/data

include $(SN_ROOT)/sw/kernels/common.mk