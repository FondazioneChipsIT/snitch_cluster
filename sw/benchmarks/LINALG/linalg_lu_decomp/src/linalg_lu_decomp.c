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
    if(use_opt == 1)
        lu_decomp_opt(core_idx, ncores, &start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec,
                      row_k, row_b, vec_write_back);
    else
        lu_decomp_naive(core_idx, ncores, &start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec);

    // Performance metrics
    total_cycles[core_idx] = end_cycle[core_idx] - start_cycle[core_idx];
    // FLOP count: (2/3*n^3) + 2*n*swap_rows_times
    flop_cycle[core_idx] = ((2.0/3.0*elems*elems*elems) + 2*elems*swap_rows_times)/(float)total_cycles[core_idx];

    snrt_cluster_hw_barrier();

    if (core_idx == 0) {

        // Mean performance values
        uint64_t mean_cycles=0;
        float mean_flop_cycle = 0.0;
        float total_flop_cycle = 0.0;

        for(uint32_t core_idx = 0; core_idx < ncores; core_idx ++){
            mean_cycles += total_cycles[core_idx];
            total_flop_cycle += flop_cycle[core_idx];
        }
        mean_cycles /= ncores;
        mean_flop_cycle = total_flop_cycle/ncores;

        printf("LU decomposition %dx%d performance\n",elems,elems);
        printf("Mean cycles: %llu\n", (unsigned long long)mean_cycles);
        printf("Mean FLOP/cycle: %f\n", mean_flop_cycle);
        printf("Total FLOP/cycle: %f\n", total_flop_cycle);

        /* print perm_vec */
        printf("perm_vec: ");
        for (uint32_t i=0;i<elems;i++) printf("%d ", perm_vec[i]);
        printf("\n");

        /* verify, will give warnings for sqrt! */
        if(verify_results == 1)
            verify_lu(orig_buf, mat, perm_vec, elems);
    }


    return 0;
}
