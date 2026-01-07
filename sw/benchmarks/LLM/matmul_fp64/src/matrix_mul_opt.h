// Luca Colombo Chips-IT 2025
// 1.2 flop at 32 elems vs 0.27, a 4.4x improvment
// 
void matrix_mul_opt(uint32_t chunk_per_core, uint32_t offset,
                    double *mat_a, double* mat_b, double *dst){

    double zero = 0.0; // Zero register

    snrt_mcycle();
    
    // Read the mat_a matrix with a 1 stride-> access a row
    snrt_ssr_loop_1d(SNRT_SSR_DM0, elems*elems, sizeof(double));

    // Read the mat_b matrix with a elems stride-> access a column (in memory the matrix is put
    // row after row)
    snrt_ssr_loop_1d(SNRT_SSR_DM1, elems, elems*sizeof(double)); 

    // Write the dst matrix with a elems stride-> access a column (in memory the matrix is put
    // row after row), only 1 element of the column per time, total chunK_per_core*elems elements
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 1, elems*sizeof(double)); 
    
    snrt_ssr_enable();
    /* Load zero into ft3, used as accumulator */
    asm volatile(
    "fld ft3, 0(%[zero])\n"
    "fld ft4, 0(%[zero])\n"
    "fld ft5, 0(%[zero])\n"
    "fld ft6, 0(%[zero])\n"
    :
    : [zero] "r"(&zero)
    : "ft3", "ft4", "ft5", "ft6");
    // Columns of mat_b, chunk_per_core times
    for(uint32_t cols = 0; cols < chunk_per_core; cols++){
        // All rows of mat_a, all for each column of mat b
        // Read all rows of mat_a
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat_a);
        
        for(uint32_t rows = 0; rows < elems; rows++){
            // As for the column the read is easy as it is only offset+rows (stored as rows)
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, mat_b + cols + offset);
            // Write stream for dst, colum indexed by offset + rows*elems (row 0,1,2 etc...)
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, dst + cols + offset + rows*(size_t)elems);
            
            asm volatile(
                "frep.o %[n_frep], 4, 0, 0 \n"  /* Repeat elems times: ft3 = ft0 (mat_a) * ft1 (mat_b) + ft3 (acc)*/
                "fmadd.d ft3, ft0, ft1, ft3\n"
                "fmadd.d ft4, ft0, ft1, ft4\n"
                "fmadd.d ft5, ft0, ft1, ft5\n"
                "fmadd.d ft6, ft0, ft1, ft6\n"
                "fadd.d ft3, ft3, ft4\n" /* Store back result in dst ft2 (dst) = ft3 (result) + ft4 (0) */
                "fadd.d ft5, ft5, ft3\n" 
                "fadd.d ft2, ft5, ft6\n" 
                "fsub.d ft3, ft3, ft3\n" // Reset accs
                "fsub.d ft4, ft4, ft4\n"
                "fsub.d ft5, ft5, ft5\n"
                "fsub.d ft6, ft6, ft6\n"
                :
                : [n_frep] "r"(elems/4 - 1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "memory");
               
        }    
    }

    snrt_ssr_disable();
    snrt_fpu_fence();
    
    snrt_mcycle();

    return;
}