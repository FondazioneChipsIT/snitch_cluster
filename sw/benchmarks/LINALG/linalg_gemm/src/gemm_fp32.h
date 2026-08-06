// Luca Colombo Chips-IT 2026
// GEMM with C = alpha* A*B + beta*C
void gemm_fp32(uint32_t chunk_per_core, uint32_t offset,
                    float alpha, float beta,
                    float *mat_a, float* mat_b, float *mat_c,
                    uint32_t m, uint32_t n, uint32_t k){

    float zero = 0.0f; // Zero register

    // Columns are assigned to the cores INTERLEAVED (core c takes the columns
    // c, c+ncores, c+2*ncores, ...) instead of in one contiguous block.
    // Reason: a column of mat_b is read with stride n*sizeof(float) = 256 B,
    // and the TCDM has 32 banks of 4 B, so a whole column lives in the single
    // bank (col % 32). With the contiguous mapping the cores 0-4, 1-5, 2-6 and
    // 3-7 end up on the same bank for the entire kernel and every access is
    // serialized 2:1. Interleaving gives every core a different bank.
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores   = snrt_cluster_compute_core_num();

    // Read the mat_a matrix with a 1 stride-> access a row
    snrt_ssr_loop_1d(SNRT_SSR_DM0, m*k, sizeof(float));

    // Read the mat_b matrix with a k stride-> access a column (in memory the matrix is put
    // row after row)
    snrt_ssr_loop_1d(SNRT_SSR_DM1, k, n*sizeof(float)); 

    // Write the dst matrix with a n stride-> access a column (in memory the matrix is put
    // row after row), only 1 element of the column per time, total chunK_per_core*n elements
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 1, n*sizeof(float)); 

    /* Load zero into the accumulators */
    asm volatile(
    "flw ft3, 0(%[zero])\n"
    "flw ft4, 0(%[zero])\n"
    "flw ft5, 0(%[zero])\n"
    "flw ft6, 0(%[zero])\n"
    "flw ft7, 0(%[alpha])\n"
    "flw ft8, 0(%[beta])\n"
    "flw ft11, 0(%[zero])\n"
    :
    : [zero] "r"(&zero), [alpha] "r"(&alpha), [beta] "r"(&beta)
    : "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "memory");

    snrt_ssr_enable();

    // Columns of mat_b, chunk_per_core times
    for(uint32_t cols = 0; cols < chunk_per_core; cols++){
        // Interleaved column owned by this core (see the note above)
        uint32_t col = core_idx + cols * ncores;

        // All rows of mat_a, all for each column of mat b
        // Read all rows of mat_a, one at a time, with stride 1, so we get a row of mat_a in each iteration
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat_a + 0);

        for(uint32_t rows = 0; rows < m; rows++){
            // As for the column the read is easy as it is only col (stored as rows)
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, mat_b + col);
            // Write stream for dst, colum indexed by col + rows*elems (row 0,1,2 etc...)
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, mat_c + col + rows*(size_t)n);

            float *ptr = mat_c + col + rows*n; // Pointer to the output element
            // Explicit assembly to avoid letting the compiler
            // use ft0 (will result in a deadlock as the 
            // SSRs are read before the computation)
            // ft9 = beta* C_val
            asm volatile(
            "flw ft9, 0(%[ptr])\n"
            "fmul.s ft9, ft8, ft9\n" 
            :: [ptr] "r"(ptr)
            : "ft9", "memory");

            asm volatile(
                "frep.o %[n_frep], 4, 0, 0 \n"  /* Repeat k/4 times: ft3 = ft0 (mat_a) * ft1 (mat_b) + ft3 (acc)*/
                "fmadd.s ft3, ft0, ft1, ft3\n"
                "fmadd.s ft4, ft0, ft1, ft4\n"
                "fmadd.s ft5, ft0, ft1, ft5\n"
                "fmadd.s ft6, ft0, ft1, ft6\n"

                "fadd.s ft3, ft4, ft3\n" /* Reduce the 4 accumulators into one */
                "fadd.s ft5, ft5, ft3\n" 
                "fadd.s ft10, ft6, ft5\n" 

                "fmadd.s ft10, ft10, ft7, ft9\n" /* Add beta * c_val, ft10 = alpha *A*B + beta * C_val */

                "fadd.s ft2, ft11, ft10\n" /* Store result in dst */

                "fsub.s ft3, ft3, ft3\n" // Reset accs
                "fsub.s ft4, ft4, ft4\n"
                "fsub.s ft5, ft5, ft5\n"
                "fsub.s ft6, ft6, ft6\n"
                : 
                : [n_frep] "r"(k/4 - 1), [dst] "r"(ptr)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "memory");
        }    

    }
    snrt_ssr_disable();
    snrt_fpu_fence();
    
    return;
}