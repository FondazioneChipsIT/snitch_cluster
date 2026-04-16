APP              := SVM_BILL
$(APP)_BUILD_DIR := $(SN_ROOT)/sw/benchmarks/ML/$(APP)/build
SRCS             := $(SN_ROOT)/sw/benchmarks/ML/$(APP)/src/$(APP).c
$(APP)_INCDIRS   := $(SN_ROOT)/sw/benchmarks/ML/$(APP)/data

include $(SN_ROOT)/sw/kernels/common.mk