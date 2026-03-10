// Luca Colombo Chips-IT 2026
/* Cholesky decompt optimized multicore version with SSR and FREP */

#include "snrt.h"
#include "data.h"
#include "ch_decomp_opt.h"

uint32_t CHECK_RESULTS = 1;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {

        size_t size = elems * elems * sizeof(float);
        mat = (float *)snrt_l1_next();
        dst = mat + size;

        snrt_dma_start_1d(mat, mat_data, size);
        snrt_dma_wait_all();

    }

    snrt_cluster_hw_barrier();

    // kernel call
    
    ch_decomp_opt(core_idx, ncores, mat, dst, elems);

    
    uint32_t err = 0;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        for(uint32_t i = 0; i < elems * elems; i++){
            if(fabsf(dst[i] - golden_L[i]) > 1e-5f){
                err ++;
            }
        }
    }

    return err;
}
