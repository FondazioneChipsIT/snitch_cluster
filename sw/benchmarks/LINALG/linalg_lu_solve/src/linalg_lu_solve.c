// Luca Colombo Chips-IT 2025
/* LU solve optimized multicore version with SSR and FREP */
// Use lu_gen.py in main folder to obtain LU matrix calculated like in
// the lu decomp app

#include "snrt.h"
#include "data.h"
// Where LU matrix is stored
#include "data_LU.h"
#include "lu_solve_opt.h"
#include "lu_solve_naive.h"

// FOR DEBUG PURPOSE
void lu_solve_serial(float *mat, uint32_t *perm, float *y_ref, float *vec, float *x_ref) {
    for (uint32_t i=0;i<elems;i++){ y_ref[i] = 0.0f; x_ref[i]=0.0f; }

    printf("=== SERIAL DEBUG FORWARD ===\n");
    for (uint32_t m = 0; m < elems; m++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < m; k++) sum += mat[(size_t)m * elems + k] * y_ref[k];
        y_ref[m] = vec[perm[m]] - sum;
        // printf("m=%u: perm=%u vec=%12.6g sum=%12.6g y[%u]=%12.6g\n", m, perm[m], vec[perm[m]], sum, m, y_ref[m]);
    }

    printf("=== SERIAL DEBUG BACKWARD ===\n");
    for (int m = (int)elems - 1; m >= 0; m--) {
        float sum = 0.0f;
        for (uint32_t k = m+1; k < elems; k++) sum += mat[(size_t)m * elems + k] * x_ref[k];
        float diag = mat[(size_t)m * elems + m];
        if (diag == 0.0f) printf("ZERO DIAG AT m=%d\n", m);
        x_ref[m] = (y_ref[m] - sum) / diag;
        // printf("m=%d: diag=%12.6g sum=%12.6g result[%d]=%12.6g\n", m, diag, sum, m, x_ref[m]);
    }
}

// Use optimized or naive version
uint32_t use_opt = 1;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {

        mat = (float *)snrt_l1_next();
        perm_vec = (uint32_t *)(mat + elems * elems);
        y = (float *)(perm_vec + elems);
        vec = y + elems;
        result = vec + elems;
        local_sum = result + elems;

        if (!mat || !perm_vec || !y || !result || !local_sum) {
            printf("Memory allocation failed!\n");
            return -1;
        }

        for (uint32_t i=0; i<elems; i++) {
            for (uint32_t j=0; j<elems; j++) {
                mat[i*elems + j] = mat_LU[i*elems + j]; // Copy the generated LU matrix by the 
                // python script in the L1 pointer
            }
        }

        for (uint32_t i=0;i<elems;i++){
            perm_vec[i] = pivots[i]; // Copy the perm
            vec[i] = i + 1.0f; // initialize the know vector
            result[i] = 0.0f;
        }
        
    }

    snrt_cluster_hw_barrier();

    // kernel call for every core as there are barriers
    if(use_opt == 1)
        lu_solve_opt(core_idx, ncores, &start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec, y, vec, result);
    else
        lu_solve_naive(&start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec, y, vec, result, local_sum);

    // DEBUG 
    //if(core_idx == 0) lu_solve_serial(mat,perm_vec,y,vec,result);
    return 0;
}
