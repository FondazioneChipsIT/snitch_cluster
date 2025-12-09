APP              := hello
$(APP)_BUILD_DIR := $(SN_ROOT)/sw/PLAY_codes/$(APP)/build
SRCS             := $(SN_ROOT)/sw/PLAY_codes/$(APP)/src/$(APP).c

include $(SN_ROOT)/sw/kernels/common.mk