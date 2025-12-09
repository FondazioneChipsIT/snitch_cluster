APP              := ssr_overhead
$(APP)_BUILD_DIR := $(SN_ROOT)/sw/a_Tests/$(APP)/build
SRCS             := $(SN_ROOT)/sw/a_Tests/$(APP)/src/ssr_overhead.c
$(APP)_INCDIRS   := $(SN_ROOT)/sw/a_Tests/$(APP)/data

include $(SN_ROOT)/sw/kernels/common.mk