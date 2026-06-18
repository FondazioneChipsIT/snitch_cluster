// Luca Colombo Chips-IT 2025
/* LU solve optimized multicore version with SSR and FREP */
// Use lu_gen.py in main folder to obtain LU matrix calculated like in
// the lu decomp app

#include "snrt.h"
#include "data.h"
// Where LU matrix is stored
#include "data_LU.h"
#include "lu_solve_naive.h"


int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {

        mat = (float *)snrt_l1_next();
        perm_vec = (uint32_t *)(mat + N * N);
        y = (float *)(perm_vec + N);
        vec = y + N;
        result = vec + N;
        local_sum = result + N;

        for (uint32_t i=0; i<N; i++) {
            for (uint32_t j=0; j<N; j++) {
                mat[i*N + j] = mat_LU[i*N + j]; // Copy the generated LU matrix by the 
                // python script in the L1 pointer
            }
        }

        for (uint32_t i=0;i<N;i++){
            perm_vec[i] = pivots[i]; // Copy the perm
            vec[i] = i + 1.0f; // initialize the know vector
            result[i] = 0.0f;
        }
        
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle(); 

    if(snrt_is_compute_core()){
        lu_solve_naive(mat, perm_vec, y, vec, result, local_sum);
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();
    
    return 0;
}
