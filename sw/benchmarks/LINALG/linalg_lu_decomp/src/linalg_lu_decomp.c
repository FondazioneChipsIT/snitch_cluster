// Luca Colombo Chips-IT 2025
/* LU optimized multicore version with SSR and FREP */

#include "snrt.h"
#include "data.h"
#include "verify_lu.h"
#include "lu_decomp_opt.h"
#include "lu_decomp_naive.h"

// Use optimized or naive version
uint32_t use_opt = 1;

// Verify results, slow and warnings for sqrt
uint32_t verify_results = 1;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {

        mat = (float *)snrt_l1_next();
        orig_buf = mat + elems * elems;
        perm_vec = (int *)(orig_buf + elems * elems);
        // elems*ncore
        vec_write_back = (float *)(perm_vec + elems);
        row_k = vec_write_back + elems*ncores;
        row_b = row_k + elems;

        if (!mat || !orig_buf || !perm_vec || !vec_write_back) {
            printf("Memory allocation failed!\n");
            return -1;
        }

        for (uint32_t i=0; i<elems; i++) {
            for (uint32_t j=0; j<elems; j++) {
                mat[i*elems + j] = ((i + j) % elems) + 1 + i;
                orig_buf[i*elems + j] = mat[i*elems + j];
            }
        }

        for (uint32_t i=0;i<elems;i++) perm_vec[i] = i;
    }

    snrt_cluster_hw_barrier();

    // kernel call
    if(snrt_is_compute_core()) {
        if(use_opt == 1)
            lu_decomp_opt(core_idx, ncores, &start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec,
                        row_k, row_b, vec_write_back);
        else
            lu_decomp_naive(core_idx, ncores, &start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec);
    }

    
    return 0;
}
