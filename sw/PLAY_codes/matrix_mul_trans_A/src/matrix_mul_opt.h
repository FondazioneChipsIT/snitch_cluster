// Luca Colombo Chips-IT 2025
//
void matrix_mul_opt(uint32_t chunk_per_core, uint32_t offset,
                    uint64_t *start_cycle, uint64_t *end_cycle,
                    double *mat_a, double* mat_b, double *dst){

    double zero = 0.0; // Zero register
    
    /* Load zero into ft4 */
    asm volatile(
    "fld ft4, 0(%[zero])\n"
    :
    : [zero] "r"(&zero)
    : "ft4");
    
    *start_cycle = snrt_mcycle();

    // Read the mat_a matrix with a 1 stride-> access a row
    snrt_ssr_loop_1d(SNRT_SSR_DM0, elems, sizeof(double));

    // Read the mat_b matrix with a elems stride-> access a column (in memory the matrix is put
    // row after row)
    snrt_ssr_loop_1d(SNRT_SSR_DM1, elems, elems*sizeof(double)); 

    // Write the dst matrix with a elems stride-> access a column (in memory the matrix is put
    // row after row), only 1 element of the column per time, total chunK_per_core*elems elements
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 1, elems*sizeof(double)); 

    
    // Columns of mat_b, chunk_per_core times
    for(uint32_t cols = 0; cols < chunk_per_core; cols++){
        // All rows of mat_a, all for each column of mat b
        for(uint32_t rows = 0; rows < elems; rows++){

            // Read all rows of mat_a
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat_a + rows*(size_t)elems);
            // As for the column the read is easy as it is only offset+rows (stored as rows)
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, mat_b + cols + offset);

            // Write stream for dst, colum indexed by offset + rows*elems (row 0,1,2 etc...)
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, dst + cols + offset + rows*(size_t)elems);
            
            /* Load zero into ft3, used as accumulator */
            asm volatile(
            "fld ft3, 0(%[zero])\n"
            :
            : [zero] "r"(&zero)
            : "ft3");
            
            snrt_ssr_enable();

            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"  /* Repeat elems times: ft3 = ft0 (mat_a) * ft1 (mat_b) + ft3 (acc)*/
                "fmadd.d ft3, ft0, ft1, ft3\n"
                "fadd.d ft2, ft3, ft4\n" /* Store back result in dst ft2 (dst) = ft3 (result) + ft4 (0) */
                :
                : [n_frep] "r"(elems - 1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "memory");

            snrt_ssr_disable();
            snrt_fpu_fence();   
        }
    }

    *end_cycle = snrt_mcycle();

    return;
}