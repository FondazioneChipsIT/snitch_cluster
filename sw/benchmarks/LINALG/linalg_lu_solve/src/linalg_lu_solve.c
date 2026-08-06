// Luca Colombo Chips-IT 2025
/* LU solve optimized multicore version with SSR and FREP */
// Use lu_gen.py in main folder to obtain LU matrix calculated like in
// the lu decomp app

#include "snrt.h"
#include "data.h"
// Where LU matrix is stored
#include "data_LU.h"
#include "lu_solve_naive.h"

uint32_t CHECK_RESULTS = 1;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {

        // Barrier in TCDM (see the note in lu_solve_naive.h). It must be
        // allocated before the layout below, so that snrt_l1_next() returns
        // memory placed after it.
        barr = (snrt_barrier_t *)snrt_l1_alloc(sizeof(snrt_barrier_t));
        barr->cnt = 0;
        barr->iteration = 0;

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

    uint32_t err = 0;
    float eps = 1e-3f;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        // Check the two equations the solver is supposed to satisfy, using the
        // LU factors already in memory: L*y = P*vec and U*result = y.
        // L is unit lower triangular, U upper triangular, both packed in mat.
        // The tolerance is relative to the magnitude of the terms being summed,
        // not to the result: these sums cancel heavily, so a residual scaled on
        // the result alone would flag a perfectly good solve.
        for (uint32_t i = 0; i < N; i++) {
            float acc = y[i];                          // L[i][i] = 1
            float mag = fabsf(y[i]);
            for (uint32_t k = 0; k < i; k++) {
                float t = mat[i*N + k] * y[k];
                acc += t;
                mag += fabsf(t);
            }

            if (fabsf(acc - vec[perm_vec[i]]) > eps * (1.0f + mag)) {
                err ++;
            }
        }

        for (uint32_t i = 0; i < N; i++) {
            float acc = 0.0f;
            float mag = 0.0f;
            for (uint32_t j = i; j < N; j++) {
                float t = mat[i*N + j] * result[j];
                acc += t;
                mag += fabsf(t);
            }

            if (fabsf(acc - y[i]) > eps * (1.0f + mag)) {
                err ++;
            }
        }
        printf("Errors: %u\n", err);
    }

    return err;
}
