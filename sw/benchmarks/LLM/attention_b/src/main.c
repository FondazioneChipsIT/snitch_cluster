// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Luca Colombo Chips-IT

#include "blas.h"
#include "dnn.h"

#include "snrt.h"

#include "data.h"

#include "flashattention.h"

// Full matrices in TCDM
float *Q_TCDM, *K_TCDM, *V_TCDM, *O_TCDM;

// Scratch buffers in TCDM
float *S_fa, *P_fa, *m_i, *m_i_prev, *l_i, *V_t;

uint32_t CHECK_RESULT = 1;

int main() {

    uint32_t core_idx = snrt_cluster_core_idx();

    uint32_t q_size  = L * D * sizeof(float);
    uint32_t kv_size = S * D * sizeof(float);

    if (snrt_is_dm_core()) {
        float *ptr = (float *)snrt_l1_next();

        Q_TCDM   = ptr; ptr += L * D;
        K_TCDM   = ptr; ptr += S * D;
        V_TCDM   = ptr; ptr += S * D;
        O_TCDM   = ptr; ptr += L * D;
        S_fa     = ptr; ptr += B_R * B_C;
        P_fa     = ptr; ptr += B_R * B_C;
        m_i      = ptr; ptr += B_R;
        m_i_prev = ptr; ptr += B_R;
        l_i      = ptr; ptr += B_R;
        V_t      = ptr; /* B_C * D */

        snrt_dma_start_1d(Q_TCDM, Q, q_size);
        snrt_dma_start_1d(K_TCDM, K, kv_size);
        snrt_dma_start_1d(V_TCDM, V, kv_size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        flashattention_2_fp32(
            Q_TCDM, K_TCDM, V_TCDM, O_TCDM,
            S_fa, P_fa, m_i, m_i_prev, l_i, V_t,
            L, S, D, B_R, B_C, BASELINE);
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(O, O_TCDM, L * D * sizeof(float));
        snrt_dma_wait_all();
    }

    snrt_global_barrier();

    uint32_t err = 0;

    if(CHECK_RESULT) {
    
        const float eps = 1e-4f;

        if (core_idx == 0) {
            uint32_t total = L * D;
            for (uint32_t i = 0; i < total; i++) {
                if (fabsf(O[i] - O_golden[i]) > eps) {
                    err++;
                }
            }
            printf("Errors: %u\n", err);
        }
    }
    
    return err;

}