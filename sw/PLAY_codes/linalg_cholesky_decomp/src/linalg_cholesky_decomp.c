// Luca Colombo Chips-IT 2026
/* Cholesky decompt optimized multicore version with SSR and FREP */

#include "snrt.h"
#include "data.h"
#include "ch_decomp_opt.h"

// Use optimized or naive version
bool use_opt = 1;


int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {

        mat = (double *)snrt_l1_next();
        dst = mat + elems * elems;

        if (!mat || !dst) {
            printf("Memory allocation failed!\n");
            return -1;
        }

        for (uint32_t i=0; i<elems; i++) {
            for (uint32_t j=0; j<elems; j++) {
                mat[i*elems + j] = ((i + j) % elems) + 1 + i;
                dst[i*elems + j] = mat[i*elems + j];
            }
        }

    }

    snrt_cluster_hw_barrier();

    // kernel call
    if(use_opt)
        ch_decomp_opt(core_idx, ncores, &start_cycle[core_idx], &end_cycle[core_idx], mat, dst, elems);
    else{

    }

    // Performance metrics
    total_cycles[core_idx] = end_cycle[core_idx] - start_cycle[core_idx];

    snrt_cluster_hw_barrier();

    if (core_idx == 0) {

        // Mean performance values
        uint64_t mean_cycles=0;

        for(uint32_t core_idx = 0; core_idx < ncores; core_idx ++){
            mean_cycles += total_cycles[core_idx];

        }
        mean_cycles /= ncores;

        printf("Cholesky decomposition %dx%d performance\n",elems,elems);
        printf("Mean cycles: %llu\n", (unsigned long long)mean_cycles);
    }
    return 0;
}
