// Luca Colombo Chips-IT 2025

// NAIVE VESION TO CHECK CORRECTNESS and PERFORMANCE BASELINE

/* 8x8 naive multicore 
# Mean cycles: 8895
# Mean FLOP/cycle: 0.045568
# Total FLOP/cycle: 0.364545
*/


/* 16x16 naive multicore 
# Mean cycles: 20582
# Mean FLOP/cycle: 0.145108
# Total FLOP/cycle: 1.160864
*/

/* 32x32 naive multicore 
# Mean cycles: 59349
# Mean FLOP/cycle: 0.385335
# Total FLOP/cycle: 3.082679
*/

/* 40x40 naive multicore 
# Matrix multiplication 40x40 performance
# Mean cycles: 92430
# Mean FLOP/cycle: 0.478920
# Total FLOP/cycle: 3.831363
*/

/*
 LU decomposition 64x64 performance
# Mean cycles: 243103
# Mean FLOP/cycle: 0.735732
# Total FLOP/cycle: 5.885853
*/

static inline void swap_rows_simple(float *mat, uint32_t n, uint32_t r1, uint32_t r2) {
    for (uint32_t j = 0; j < n; j++) {
        float tmp = mat[r1*n + j];
        mat[r1*n + j] = mat[r2*n + j];
        mat[r2*n + j] = tmp;
    }
}


void lu_decomp_naive(uint32_t core_idx ,uint32_t ncores, uint64_t *start_cycle, uint64_t *end_cycle, 
                            float *mat, int *perm) {

    for (uint32_t k = 0; k < elems; k++) {

        if(core_idx == 0){
            /* pivot selection on current column */
            uint32_t row_max = k;
            float max_val = fabsf(mat[k * elems + k]);
            
            for (uint32_t m = k + 1; m < elems; m++) {
                float cur = fabsf(mat[m * elems + k]);
                if (cur > max_val) { 
                    row_max = m; 
                    max_val = cur;
                }
            }

            // Check for zero pivot
            if (max_val < 1e-12f) {
                printf("ERROR: zero pivot at k=%u\n", (unsigned)k);
                *end_cycle = snrt_mcycle();
                return;
            }
            // Swap rows if needed
            if (row_max != k) {
                swap_rows_simple(mat, elems, k, row_max);
                int tmp = perm[k];
                perm[k] = perm[row_max];
                perm[row_max] = tmp;
                // update swap count
                swap_rows_times++;
            }
        }
    
        snrt_partial_barrier(&barr, 8);
        float pivot = mat[k * elems + k];
        float p_inv = 1.0f / pivot;

    

        /* Gaussian Elimination */
        int start;
        int block;
        int left;
        int end;

        block = (elems - (k + 1)) / ncores;
        left = (elems - (k + 1)) % ncores;
        start = core_idx * block + (core_idx < left ? core_idx : left) + (k + 1);
        end = start + block + (core_idx < left ? 1 : 0);

        for (int m = start; m < end; m++) {
            float factor;

            factor = mat[m * elems + k] * p_inv;
            mat[m * elems + k] = factor;

            for (int n = k + 1; n < elems; n++)
                mat[m * elems + n] -= factor * mat[k * elems + n];
        }
    

        snrt_partial_barrier(&barr, 8);
    }

}
