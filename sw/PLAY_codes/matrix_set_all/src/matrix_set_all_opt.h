// Luca Colombo Chips-IT 2025

void matrix_set_all_opt(uint32_t chunk_per_core, uint32_t offset, 
                        uint64_t *start_cycle, uint64_t *end_cycle, 
                        double *mat_a, double *val){

    // Perform operations on the matrix like if it was a vector, rows
    double zero = 0.0;

    *start_cycle = snrt_mcycle();

    // Setup the 1d loop with ssr (tell which streams to use, the size and the size of
    // the elements)
    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(double));
    // Write to ft0
    snrt_ssr_write(SNRT_SSR_DM0, SNRT_SSR_1D, mat_a + offset); //ft0->mat_a
    
    // Load the scalar value val in ft3 register and zero in ft4
    asm volatile(
        "fld ft3, 0(%[val])\n"
        "fld ft4, 0(%[zero])\n"
        :                   // Outputs
        : [zero] "r"(zero), [val] "r"(val)    // Inputs 
        : "ft3");           // Clobber list

    // Enable the SSRs
    snrt_ssr_enable();
    
    // Assembly code to add ft3 (val) to ft4 (0) and store in ft1 (vec)
    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fadd.d ft0, ft3, ft4\n"
        :
        : [ n_frep ] "r"(chunk_per_core - 1)
        : "ft0", "ft3", "ft4", "memory");

    // Disable SSRs
    snrt_ssr_disable();
    // Fence for FPU syncronization
    snrt_fpu_fence();

    *end_cycle = snrt_mcycle();

    return;
}