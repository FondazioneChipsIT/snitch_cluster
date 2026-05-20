#!/usr/bin/env bash

# File: sw/benchmarks/auto_benchmark.sh

set -euo pipefail

RESULTS_DIR="sw/benchmarks/results"
mkdir -p "${RESULTS_DIR}"
source chips-it-setup.sh

# Run a compilation to ensure all benchmarks are built before we start running them
make sw

# ============================================================
# Hardcoded benchmark list
# ============================================================

BENCHMARKS=(
    # DSP benchmarks
    "sw/benchmarks/DSP/FFT/build/FFT.elf"
    "sw/benchmarks/DSP/DWT/build/DWT.elf"
    "sw/benchmarks/DSP/FIR/build/FIR.elf"
    # Linear algebra benchmarks
    "sw/benchmarks/LINALG/linalg_cholesky_decomp/build/linalg_cholesky_decomp.elf"
    "sw/benchmarks/LINALG/linalg_gemm/build/linalg_gemm.elf"
    "sw/benchmarks/LINALG/linalg_gemv/build/linalg_gemv.elf"
    "sw/benchmarks/LINALG/linalg_gemvt/build/linalg_gemvt.elf"
    "sw/benchmarks/LINALG/linalg_lu_decomp/build/linalg_lu_decomp.elf"
    "sw/benchmarks/LINALG/linalg_lu_solve/build/linalg_lu_solve.elf"
    "sw/benchmarks/LINALG/linalg_svd_jacobi/build/linalg_svd_jacobi.elf"
    # LLM benchmarks
    "sw/benchmarks/LLM/matmul_fp32/build/matmul_fp32.elf"
    "sw/benchmarks/LLM/attention_b/build/attention_b.elf"
    "sw/benchmarks/LLM/encoder/build/encoder.elf"
    "sw/benchmarks/LLM/gelu_b/build/gelu_b.elf"
    "sw/benchmarks/LLM/layernorm_b/build/layernorm_b.elf"
    "sw/benchmarks/LLM/residual/build/residual.elf"
    "sw/benchmarks/LLM/softmax_b/build/softmax_b.elf"
    # ML benchmarks
    "sw/benchmarks/ML/CONV3x3/build/CONV3x3.elf"
    "sw/benchmarks/ML/kmeans_b/build/kmeans_b.elf"
    "sw/benchmarks/ML/RANDOM_FOREST/build/RANDOM_FOREST.elf"
    "sw/benchmarks/ML/SVM_BILL/build/SVM_BILL.elf"
    "sw/benchmarks/ML/SVM_CANC/build/SVM_CANC.elf"
)

# ============================================================

for ELF_PATH in "${BENCHMARKS[@]}"; do

    echo "========================================"
    echo "Running benchmark: ${ELF_PATH}"
    echo "========================================"

    if [ ! -f "${ELF_PATH}" ]; then
        echo "ERROR: ELF not found: ${ELF_PATH}"
        continue
    fi

    BENCH_NAME=$(basename "${ELF_PATH}" .elf)

    # Clean previous outputs
    rm -rf logs traces

    echo "[1/5] Running simulation..."

    VSIM=$(find . -name "snitch_cluster.vsim" | head -n 1)

    if [ -z "${VSIM}" ]; then
        echo "ERROR: snitch_cluster.vsim not found"
        exit 1
    fi

    echo "Using simulator: ${VSIM}"

    "${VSIM}" "${ELF_PATH}"

    echo "[2/5] Generating traces..."

    make traces SIM_DIR=. -j

    echo "[3/5] Finding core 0 trace..."

    TRACE_FILE="logs/trace_hart_00000.txt"

    if [ -z "${TRACE_FILE}" ]; then
        echo "ERROR: Could not find core 0 trace file"
        continue
    fi

    echo "Using trace: ${TRACE_FILE}"

    echo "[4/5] Extracting mcycle PCs..."

    MCYCLE_LINES=$(grep "mcycle" "${TRACE_FILE}" | head -n 2)

    COUNT=$(echo "${MCYCLE_LINES}" | wc -l)

    if [ "${COUNT}" -lt 2 ]; then
        echo "ERROR: Could not find two mcycle accesses"
        continue
    fi

    # Take first hex value in line as PC
    START_PC=$(echo "${MCYCLE_LINES}" | sed -n '1p' | grep -o '0x[0-9a-fA-F]\+' | head -n 1)
    END_PC=$(echo "${MCYCLE_LINES}" | sed -n '2p' | grep -o '0x[0-9a-fA-F]\+' | head -n 1)

    if [ -z "${START_PC}" ] || [ -z "${END_PC}" ]; then
        echo "ERROR: Failed to extract PCs"
        continue
    fi

    echo "Start PC: ${START_PC}"
    echo "End PC:   ${END_PC}"

    echo "[5/5] Running flop parser..."

    OUTPUT_FILE="${RESULTS_DIR}/${BENCH_NAME}.txt"

    python sw/benchmarks/pulp_cluster_flop_parsing.py \
        --folder logs \
        --start "${START_PC}" \
        --end "${END_PC}" \
        > "${OUTPUT_FILE}"

    echo "Saved results to ${OUTPUT_FILE}"
    echo

done

echo "All benchmarks completed."