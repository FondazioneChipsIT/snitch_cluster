// Luca Colombo Chips-IT 2025
// Optimized version with SSR and FREP

#include "matrix_swap_rows_opt2.h"

// in case we need the simple version, for elems <32
static inline void swap_rows_simple(float *mat, uint32_t n, uint32_t r1, uint32_t r2);

// The barrier must NOT be a plain global: globals are linked into L3 (the
// linker script only has the DRAM region), so every snrt_partial_barrier would
// spin on a DRAM address at ~60 cycles per access. It is allocated in TCDM by
// the DM core in main instead, and this global only holds the pointer.
snrt_barrier_t *barr;

void lu_decomp_opt(uint32_t core_idx ,uint32_t ncores, uint64_t *start_cycle, uint64_t *end_cycle, 
                        float *mat, int *perm, 
                        float *row_k, float *row_b, float *vec_write_back) {

    // Local copy: otherwise the global pointer is re-read from DRAM after
    // every barrier call (the call writes memory, so it cannot be cached)
    snrt_barrier_t *bar_p = barr;


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
    uint32_t vec_offset = core_idx * cols;

    // Zero pivot threshold
    float eps = 1e-15f;

    float zero =  0.0f;

    /* Load zero into ft5 */
    asm volatile(
    "flw ft5, 0(%[zero])\n"
    :
    : [zero] "r"(&zero)
    : "ft5");


    for (uint32_t k = 0; k < cols; k++) {

        // Only core 0 does pivot selection and row swap, not possible to parallelize
        if(core_idx == 0){

            /* pivot selection on current column */
            row_max = k;
            max_val = fabsf(mat[k * cols + k]);
            
            // Could be optimized with SSRs? 
            // but overhead probably too high for single column
            // and difficult to implement efficiently
            for (uint32_t m = k + 1; m < rows; m++) {
                float cur = fabsf(mat[m * cols + k]);
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
                matrix_swap_rows_opt2(cols, 0, mat, row_k, row_b, k, row_max);

                // Or simple version if needed, rows<32
                // swap_rows_simple(mat, rows, k, row_max);

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
        snrt_partial_barrier(bar_p, 8);

        // Each core has its own copy of the pivot row
        float pivot = mat[k * cols + k];
        float p_inv = 1.0f / pivot;

    
    
        // The rows to eliminate are m in [k+1, rows): THOSE are what gets split
        // among the cores. Before, the split used (cols - (k+1)), so with
        // rows != cols (here 64x32) the rows from cols on were never eliminated.
        left = (rows - (k + 1)) % ncores;
        base = (rows - (k + 1)) / ncores;
        chunk_per_core = base + (core_idx < left ? 1 : 0);
        offset = (k + 1) + core_idx * base + (core_idx < left ? core_idx : left);
        end = offset + chunk_per_core;

        // Inside a row we update the columns after k, so the stream is
        // cols-(k+1) elements long (it used to be rows-(k+1), which ran past
        // the end of the row and into the next one).
        uint32_t col_chunk = cols - (k + 1);
        uint32_t col_offset = k + 1;

        for (int m = offset; m < end; m++) {

            float factor;

            // Same as before, cannot be optimized with SSRs
            factor = mat[m * cols + k] * p_inv;
            mat[m * cols + k] = factor;

            // On the last column there is nothing left to update
            if (col_chunk == 0) continue;

                /* Load factor into ft3 */
            asm volatile(
            "flw ft3, 0(%[factor])\n"
            :
            : [factor] "r"(&factor)
            : "ft3");

            // 3 SSR to read matrix rows and write vec
            snrt_ssr_loop_1d(SNRT_SSR_DM0, col_chunk, sizeof(float));
            snrt_ssr_loop_1d(SNRT_SSR_DM1, col_chunk, sizeof(float));
            snrt_ssr_loop_1d(SNRT_SSR_DM2, col_chunk, sizeof(float));

            // read the matrix, use a buffer to store results
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat + k * cols + col_offset); // pivot row
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, mat + m * cols + col_offset); // current row
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, mat + m * cols + col_offset); // write back

            snrt_ssr_enable();

            asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"  // repeat the next instruction
            // fnmsub, NOT fmsub: fmsub.s rd,rs1,rs2,rs3 computes rs1*rs2 - rs3,
            // i.e. the opposite sign of what is needed here. fnmsub.s computes
            // -(rs1*rs2) + rs3, that is ft1 - ft0*ft3.
            "fnmsub.s ft2, ft0, ft3, ft1\n" // ft2 = ft1 (current row) - ft0 (pivot row) * ft3 (factor)
            :
            : [n_frep] "r"(col_chunk - 1)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "memory");


            snrt_ssr_disable();
            snrt_fpu_fence();
        }

        // Barrier: core 0 must not start the pivot search of column k+1 (it
        // reads that column and may swap two whole rows) while the other cores
        // are still writing their rows for column k. Without it the loop had
        // only one barrier per iteration and the pivot could be chosen on
        // half-updated data.
        snrt_partial_barrier(bar_p, 8);

    }
    snrt_fpu_fence();
    return;
}

    


