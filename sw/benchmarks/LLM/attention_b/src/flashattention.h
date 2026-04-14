// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Luca Colombo Chips-IT
//
// Flash Attention 2 FP32 kernel — no DMA, no struct.
// All pointers must already live in TCDM when this function is called.
//
// NOTE: parameter names are prefixed with fa_ to avoid collisions with
// the #define macros (L, S, D, B_R, B_C, BASELINE) coming from data.h.

#pragma once

#include "fastexpf.h"
#include <math.h>
#include "snrt.h"
#include "blas.h"

snrt_barrier_t barr2;

/**
 * @brief Flash Attention 2 compute kernel (FP32).
 *
 * @param Q_TCDM      Full Q matrix in TCDM  (fa_L x fa_d)
 * @param K_TCDM      Full K matrix in TCDM  (fa_S x fa_d)
 * @param V_TCDM      Full V matrix in TCDM  (fa_S x fa_d)
 * @param O_TCDM      Full O matrix in TCDM  (fa_L x fa_d) — written in place
 * @param S_fa        Scratch buffer (fa_Br x fa_Bc)
 * @param P_fa        Scratch buffer (fa_Br x fa_Bc)
 * @param m_i         Scratch buffer (fa_Br) — running row max
 * @param m_i_prev    Scratch buffer (fa_Br) — previous row max
 * @param l_i         Scratch buffer (fa_Br) — running row sum
 * @param V_t         Scratch buffer (fa_Bc x fa_d), optimised path only
 * @param fa_L        Sequence length  (rows of Q / O)
 * @param fa_S        Key/value length (rows of K / V)
 * @param fa_d        Head dimension
 * @param fa_Br       Row block size
 * @param fa_Bc       Column block size
 * @param fa_baseline 1 = naive GEMM, 0 = SSR-optimised GEMM
 */
static inline void flashattention_2_fp32(
    float *Q_TCDM, float *K_TCDM, float *V_TCDM, float *O_TCDM,
    float *S_fa,   float *P_fa,
    float *m_i,    float *m_i_prev, float *l_i,
    float *V_t,
    uint32_t fa_L, uint32_t fa_S, uint32_t fa_d,
    uint32_t fa_Br, uint32_t fa_Bc,
    uint32_t fa_baseline)
{
    // Problems using the vfpk instructions with the 32-bit Snitch
    gemm_fp_t gemm_implementation = gemm_fp32_naive_unrolled;

    sc_st_gemm_args_t gemm_args;
    gemm_args.prec      = FP32;
    gemm_args.setup_ssr = fa_baseline ? 0 : 1;
    gemm_args.transa    = 0;
    gemm_args.transb    = 1;
    gemm_args.m         = fa_Br;
    gemm_args.alpha     = 1;

    uint32_t compute_id = snrt_cluster_core_idx();
    uint32_t num_cores  = snrt_cluster_compute_core_num();

    uint32_t T_r = fa_L / fa_Br;  // number of row blocks
    uint32_t T_c = fa_S / fa_Bc;  // number of column blocks

    // Row distribution across cores (same as original)
    uint32_t rows_per_core = fa_Br / num_cores;
    uint32_t start_row     = rows_per_core * compute_id;
    char is_last_core      = (compute_id == num_cores - 1);
    uint32_t end_row       = is_last_core ? fa_Br : start_row + rows_per_core;

    float shifted_exp;
    float row_sum;

    // ── Outer loop: row blocks of Q ─────────────────────────────────────────
    for (int t_r = 0; t_r < (int)T_r; t_r++) {

        float *Q_fa = &Q_TCDM[t_r * fa_Br * fa_d];
        float *O_fa = &O_TCDM[t_r * fa_Br * fa_d];

        // Initialize m_i, m_i_prev, l_i
        for (int row_idx = start_row; row_idx < (int)end_row; row_idx++) {
            m_i[row_idx]      = -(float)INFINITY;
            m_i_prev[row_idx] = -(float)INFINITY;
            l_i[row_idx]      = 0.0f;
        }

        snrt_partial_barrier(&barr2, 8);

        // ── Inner loop: column blocks of K / V ──────────────────────────────
        for (int t_c = 0; t_c < (int)T_c; t_c++) {

            float *K_fa = &K_TCDM[t_c * fa_Bc * fa_d];
            float *V_fa = &V_TCDM[t_c * fa_Bc * fa_d];

            

            // S = Q * K^T   (fa_Br x fa_Bc)
            gemm_args.n   = fa_Bc;
            gemm_args.k   = fa_d;
            gemm_args.a   = Q_fa;
            gemm_args.lda = fa_d;
            gemm_args.b   = K_fa;
            gemm_args.ldb = fa_d;
            gemm_args.beta = 0;
            gemm_args.c   = S_fa;
            gemm_args.ldc = fa_Bc;
            sc_st_gemm(gemm_implementation, &gemm_args);

            snrt_partial_barrier(&barr2, 8);

            // Per-row softmax update with running statistics
            for (int row_idx = start_row; row_idx < (int)end_row; row_idx++) {
                m_i_prev[row_idx] = m_i[row_idx];
                row_sum = 0.0f;

                // New row max
                for (int col_idx = 0; col_idx < (int)fa_Bc; col_idx++) {
                    float val = S_fa[row_idx * fa_Bc + col_idx];
                    if (val > m_i[row_idx]) m_i[row_idx] = val;
                }

                // Compute P = exp(S - m_i) and accumulate row_sum
                for (int col_idx = 0; col_idx < (int)fa_Bc; col_idx++) {
                    P_fa[row_idx * fa_Bc + col_idx] =
                        fast_expf(S_fa[row_idx * fa_Bc + col_idx] - m_i[row_idx]);
                    row_sum += P_fa[row_idx * fa_Bc + col_idx];
                }

                // Update running normaliser l_i
                shifted_exp = fast_expf(m_i_prev[row_idx] - m_i[row_idx]);
                if (t_c != 0) {
                    l_i[row_idx] = l_i[row_idx] * shifted_exp + row_sum;
                } else {
                    l_i[row_idx] = row_sum;
                }

                // Rescale O from previous iteration
                if (t_c != 0) {
                    for (int col_idx = 0; col_idx < (int)fa_d; col_idx++) {
                        O_fa[row_idx * fa_d + col_idx] /= shifted_exp;
                    }
                }
            }

            snrt_partial_barrier(&barr2, 8);

            // O += P * V   (fa_Br x fa_d)
            uint32_t beta = (t_c == 0) ? 0 : 1;

            if (fa_baseline) {
                gemm_args.n      = fa_d;
                gemm_args.k      = fa_Bc;
                gemm_args.transb = 0;
                gemm_args.a      = P_fa;
                gemm_args.lda    = fa_Bc;
                gemm_args.b      = V_fa;
                gemm_args.ldb    = fa_d;
                gemm_args.beta   = beta;
                gemm_args.c      = O_fa;
                gemm_args.ldc    = fa_d;
                sc_st_gemm(gemm_implementation, &gemm_args);
                gemm_args.transb = 1;
            } else {
                // Optimised kernel expects A * B^T → transpose V first
                transpose_kernel(FP32, V_fa, V_t, fa_Bc, fa_d, fa_baseline);

                gemm_args.n    = fa_d;
                gemm_args.k    = fa_Bc;
                gemm_args.a    = P_fa;
                gemm_args.lda  = fa_Bc;
                gemm_args.b    = V_t;
                gemm_args.ldb  = fa_Bc;
                gemm_args.beta = beta;
                gemm_args.c    = O_fa;
                gemm_args.ldc  = fa_d;
                sc_st_gemm(gemm_implementation, &gemm_args);
            }

            snrt_partial_barrier(&barr2, 8);

        }  // end T_c

        // Final rescaling: O_i = diag(l_i)^-1 * O_i
        for (int row_idx = start_row; row_idx < (int)end_row; row_idx++) {
            for (int col_idx = 0; col_idx < (int)fa_d; col_idx++) {
                O_fa[row_idx * fa_d + col_idx] /= l_i[row_idx];
            }
        }
        snrt_fpu_fence();
        snrt_partial_barrier(&barr2, 8);

    }  // end T_r
}