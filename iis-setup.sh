#!/usr/bin/env bash
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Define environment variables
export CC=
export CXX=
export SN_BENDER=bender
export SN_VCS_SEPP=
export SN_VERILATOR_SEPP=
export SN_QUESTA_SEPP=
export SN_LLVM_BINROOT=/opt/riscv/snitch-llvm-15.0.0-snitch-0.2.0/bin

# Create Python virtual environment with required packages
#python3.11 -m venv .venv
#source .venv/bin/activate
#pip install -r requirements.txt
# Install local packages in editable mode and unpack packages in a
# local temporary directory which can be safely cleaned after installation.
# Also protects against "No space left on device" errors
# occurring when the /tmp folder is filled by other processes.
#mkdir tmp
#TMPDIR=tmp pip install -e .[all]
#rm -rf tmp

# Add simulator binaries to PATH
export PATH=$PWD/target/sim/build/bin:$PATH
