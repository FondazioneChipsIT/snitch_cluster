APP              := vector_memcpy
$(APP)_BUILD_DIR := $(SN_ROOT)/sw/PLAY_codes/$(APP)/build
SRCS             := $(SN_ROOT)/sw/PLAY_codes/$(APP)/src/$(APP).c
$(APP)_INCDIRS   := $(SN_ROOT)/sw/PLAY_codes/$(APP)/data

include $(SN_ROOT)/sw/kernels/common.mk