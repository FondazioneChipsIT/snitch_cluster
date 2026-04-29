// Luca Colombo Chips-IT 2026
/* Cholesky decompt optimized multicore version with SSR and FREP */

#include "snrt.h"
#include "data.h"
#include "svd_jacobi_opt.h"

uint32_t CHECK_RESULTS = 1;


int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();
    uint32_t tot_elems = M * N;

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {
        mat = (float *)snrt_l1_next();
        mat_V = mat + tot_elems;
        vec_S = mat_V + N*N;
        local_max = vec_S + M;

        float zero[ncores];
        for (uint32_t i = 0; i < ncores; i++) {
            zero[i] = 0.0f;
        }

        size_t size = tot_elems * sizeof(float);
        snrt_dma_start_1d(mat, mat_data, size);
        snrt_dma_start_1d(local_max, zero, ncores * sizeof(float));
        snrt_dma_wait_all();

        for(uint32_t i = 0; i < N; i++){
            for(uint32_t j = 0; j < N; j++){
                mat_V[i * N + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();
    // kernel call
    if(snrt_is_compute_core()){
        svd_jacobi_opt(mat, mat_V, vec_S, M, N);
    }
    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;
    float eps = 1e-5f;

    if (core_idx == 0 && CHECK_RESULTS) {
        float sum_sq = 0.0f, golden_sum_sq = 0.0f;
        for(uint32_t i = 0; i < K; i++){
            sum_sq        += vec_S[i]    * vec_S[i];
            golden_sum_sq += golden_S[i] * golden_S[i];
        }
        if(fabsf(sum_sq - golden_sum_sq) > eps) err++;
    }
    

    return err;
}
