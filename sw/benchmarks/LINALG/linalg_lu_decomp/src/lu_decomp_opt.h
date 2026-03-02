// Luca Colombo Chips-IT 2025
// Optimized version with SSR and FREP


/* fully optimized version with SSR and FREP 
# LU decomposition 8x8 performance
# Mean cycles: 9060
# Mean FLOP/cycle: 0.044736
# Total FLOP/cycle: 0.357887
*/


/* fully optimized version with SSR and FREP 
# LU decomposition 16x16 performance
# Mean cycles: 19926
# Mean FLOP/cycle: 0.149884
# Total FLOP/cycle: 1.199074
*/

/* fully optimized version with SSR and FREP 
# LU decomposition 32x32 performance
# Mean cycles: 52049
# Mean FLOP/cycle: 0.439377
# Total FLOP/cycle: 3.515013

*/

/* fully optimized version with SSR and FREP 
 LU decomposition 40x40 performance
# Mean cycles: 74326
# Mean FLOP/cycle: 0.595574
# Total FLOP/cycle: 4.764588
*/


/*
# LU decomposition 64x64 performance
# Mean cycles: 172504
# Mean FLOP/cycle: 1.036836
# Total FLOP/cycle: 8.294684
*/

// Use optimized row swap with SSR and FREP 
// FOR 32x32 MATRIX we do not get a real benefit from this optimization
// so can roll back to naive swap if needed
// for 40x40 or larger matrices this is beneficial, slightly better than naive
#include "matrix_swap_rows_opt2.h"

// in case we need the simple version, for elems <32
static inline void swap_rows_simple(float *mat, uint32_t n, uint32_t r1, uint32_t r2);

snrt_barrier_t barr;

void lu_decomp_opt(uint32_t core_idx ,uint32_t ncores, uint64_t *start_cycle, uint64_t *end_cycle, 
                        float *mat, int *perm, 
                        float *row_k, float *row_b, float *vec_write_back) {


    // For pivot selection
    uint32_t n_rep;
    uint32_t row_max;
    float max_val;

    // Used in row distribution
    uint32_t offset;
    uint32_t chunk_per_core;
    uint32_t base;
    uint32_t remainder;
    uint32_t left;
    uint32_t end;
    uint32_t rows_left;
    uint32_t chunk_real;

    // Used for vec_write_back offset
    uint32_t vec_offset = core_idx * elems;

    // Zero pivot threshold
    float eps = 1e-15f;

    float zero =  0.0f;

    /* Load zero into ft5 */
    asm volatile(
    "flw ft5, 0(%[zero])\n"
    :
    : [zero] "r"(&zero)
    : "ft5");

    // Start cycle count
    *start_cycle = snrt_mcycle();

    for (uint32_t k = 0; k < elems; k++) {

        // Only core 0 does pivot selection and row swap, not possible to parallelize
        if(core_idx == 0){

            /* pivot selection on current column */
            row_max = k;
            max_val = fabsf(mat[k * elems + k]);
            
            // Could be optimized with SSRs? 
            // but overhead probably too high for single column
            // and difficult to implement efficiently
            for (uint32_t m = k + 1; m < elems; m++) {
                float cur = fabsf(mat[m * elems + k]);
                if (cur > max_val) { 
                    // This line cannot be optimized with SSRs
                    row_max = m; 
                    // This can be optimized with SSRs
                    max_val = cur;
                }
            }

            // Check for zero pivot
            if (max_val < eps) {
                printf("ERROR: zero pivot at k=%u\n", (unsigned)k);
                *end_cycle = snrt_mcycle();
                return;
            }
            // Swap rows if needed
            if (row_max != k) {

                // Optimized row swap with SSR and FREP
                matrix_swap_rows_opt2(elems, 0, mat, row_k, row_b, k, row_max);

                // Or simple version if needed, elems<32
                // swap_rows_simple(mat, elems, k, row_max);

                // This update cannot be optimized with SSRs, as the overhead at the start
                // and the need to change between read and write mode would be too high.
                int tmp = perm[k];
                perm[k] = perm[row_max];
                perm[row_max] = tmp;
                // update swap count, for performance metrics
                swap_rows_times++;
            }

        }
        
        // cores need to wait for the pivot row to be ready
        snrt_partial_barrier(&barr, 8);

        // Each core has its own copy of the pivot row
        float pivot = mat[k * elems + k];
        float p_inv = 1.0f / pivot;

    
    
        // for columns after k, distribute rows among cores
        left = (elems - (k + 1)) % ncores;
        base = (elems - (k + 1)) / ncores;
        chunk_per_core = base + (core_idx < left ? 1 : 0);
        offset = (k + 1) + core_idx * base + (core_idx < left ? core_idx : left);
        end = offset + chunk_per_core;

        // for rows, we must add an offset of (k+1)
        // so total rows to consider is elems-(k+1)
        uint32_t row_chunk = elems - (k + 1);
        uint32_t row_offset = k + 1;

        for (int m = offset; m < end; m++) {

            float factor;

            // Same as before, cannot be optimized with SSRs
            factor = mat[m * elems + k] * p_inv;
            mat[m * elems + k] = factor;

                /* Load factor into ft3 */
            asm volatile(
            "flw ft3, 0(%[factor])\n"
            :
            : [factor] "r"(&factor)
            : "ft3");

            // 3 SSR to read matrix rows and write vec
            snrt_ssr_loop_1d(SNRT_SSR_DM0, row_chunk, sizeof(float));
            snrt_ssr_loop_1d(SNRT_SSR_DM1, row_chunk, sizeof(float));
            snrt_ssr_loop_1d(SNRT_SSR_DM2, row_chunk, sizeof(float));

            // read the matrix, use a buffer to store results
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat + k * elems + row_offset); // pivot row
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, mat + m * elems + row_offset); // current row
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, mat + m * elems + row_offset); // write back
            
            snrt_ssr_enable();

            asm volatile(
            "frep.o %[n_frep], 2, 0, 0 \n"  // repeat the next 2 instructions
            "fmul.s ft4, ft0, ft3\n" // ft4 = ft0 (pivot row) * ft3 (factor)
            "fsub.s ft2, ft1, ft4\n" // ft2 = ft1 (current row) - ft4 (result)
            :
            : [n_frep] "r"(row_chunk - 1)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "memory");


            snrt_ssr_disable();
            snrt_fpu_fence();
        }

    

    }
        
    *end_cycle = snrt_mcycle();
}

    


