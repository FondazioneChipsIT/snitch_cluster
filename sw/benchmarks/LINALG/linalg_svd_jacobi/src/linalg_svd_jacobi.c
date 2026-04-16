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
        vec_S = mat_V + tot_elems;
        local_max = vec_S + M;

        size_t size = tot_elems * sizeof(float);
        snrt_dma_start_1d(mat, mat_data, size);
        snrt_dma_wait_all();

        for(uint32_t i = 0; i < tot_elems; i++){
            mat_V[i] = 1.0f;
        }
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();
    // kernel call
    if(snrt_is_compute_core()){
        svd_jacobi_opt(mat, mat_V, vec_S, M);
    }
    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;
    float eps = 1e-4f;

    if (core_idx == 0 && CHECK_RESULTS) {
        for(uint32_t i = 0; i < M*N; i++){
            if(fabsf(mat_V[i] - golden_V[i]) > eps || fabsf(mat[i] - golden_U[i]) > eps || fabsf(vec_S[i] - golden_S[i]) > eps){
                err ++;
            }
        }
    }

    return err;
}
