// Luca Colombo Chips-IT 2026
/* Cholesky decompt optimized multicore version with SSR and FREP */

#include "snrt.h"
#include "data.h"
#include "ch_decomp_opt.h"

uint32_t CHECK_RESULTS = 1;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();
    uint32_t tot_elems = elems * elems;

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {
        mat = (float *)snrt_l1_next();
        dst = mat + tot_elems;

        size_t size = tot_elems * sizeof(float);
        snrt_dma_start_1d(mat, mat_data, size);
        snrt_dma_wait_all();

        for(uint32_t i = 0; i < tot_elems; i++){
            dst[i] = 0.0f;
        }
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();
    // kernel call
    if(snrt_is_compute_core()){
        cholesky_opt(core_idx, ncores, mat, dst, elems);
    }
    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        for(uint32_t i = 0; i < tot_elems; i++){
            if(fabsf(dst[i] - golden_L[i]) > 1e-5f){
                err ++;
                break;
            }
        }
    }

    return err;
}
