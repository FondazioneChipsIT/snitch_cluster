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
void lu_solve_serial(double *mat, uint32_t *perm, double *y_ref, double *vec, double *x_ref) {
    for (uint32_t i=0;i<elems;i++){ y_ref[i] = 0.0; x_ref[i]=0.0; }

    printf("=== SERIAL DEBUG FORWARD ===\n");
    for (uint32_t m = 0; m < elems; m++) {
        double sum = 0.0;
        for (uint32_t k = 0; k < m; k++) sum += mat[(size_t)m * elems + k] * y_ref[k];
        y_ref[m] = vec[perm[m]] - sum;
        printf("m=%u: perm=%u vec=%12.6g sum=%12.6g y[%u]=%12.6g\n", m, perm[m], vec[perm[m]], sum, m, y_ref[m]);
    }

    printf("=== SERIAL DEBUG BACKWARD ===\n");
    for (int m = (int)elems - 1; m >= 0; m--) {
        double sum = 0.0;
        for (uint32_t k = m+1; k < elems; k++) sum += mat[(size_t)m * elems + k] * x_ref[k];
        double diag = mat[(size_t)m * elems + m];
        if (diag == 0.0) printf("ZERO DIAG AT m=%d\n", m);
        x_ref[m] = (y_ref[m] - sum) / diag;
        printf("m=%d: diag=%12.6g sum=%12.6g result[%d]=%12.6g\n", m, diag, sum, m, x_ref[m]);
    }
}

// Use optimized or naive version
bool use_opt = 1;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {

        mat = (double *)snrt_l1_next();
        perm_vec = (uint32_t *)(mat + elems * elems);
        y = (double *)(perm_vec + elems);
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
            vec[i] = i + 1.0; // initialize the know vector
            result[i] = 0.0;
        }
        
    }

    snrt_cluster_hw_barrier();

    // kernel call for every core as there are barriers
    if(use_opt)
        lu_solve_opt(core_idx, ncores, &start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec, y, vec, result);
    else
        lu_solve_naive(&start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec, y, vec, result, local_sum);

    // DEBUG 
    //if(core_idx == 0) lu_solve_serial(mat,perm_vec,y,vec,result);

    // Performance metrics
    total_cycles[core_idx] = end_cycle[core_idx] - start_cycle[core_idx];
    flop_cycle[core_idx] = (elems * (2.0 * elems - 1.0)) / (double) total_cycles[core_idx];
   
   

    snrt_cluster_hw_barrier();

    if (core_idx == 0) {

        // Mean performance values
        uint64_t mean_cycles=0;
        double mean_flop_cycle = 0.0;
        double total_flop_cycle = 0.0;

        for(uint32_t core_idx = 0; core_idx < ncores; core_idx ++){
            mean_cycles += total_cycles[core_idx];
            total_flop_cycle += flop_cycle[core_idx];
        }
        mean_cycles /= ncores;
        mean_flop_cycle = total_flop_cycle/ncores;

        printf("Lu solve %dx%d performance\n",elems,elems);
        printf("Mean cycles: %llu\n", (unsigned long long)mean_cycles);
        printf("Mean FLOP/cycle: %f\n", mean_flop_cycle);
        printf("Total FLOP/cycle: %f\n", total_flop_cycle);

        /* print result
        printf("Result: ");
        for (uint32_t i=0;i<elems;i++) printf("%f ", result[i]);
        printf(".\n");*/

    }


    return 0;
}
