// Luca Colombo Chips-IT 2025

void vect_sub_opt(uint32_t chunk_per_core, uint32_t offset, uint64_t *start_cycle,uint64_t *end_cycle, 
    double *src_a, double *src_b, double *dst){

    *start_cycle = snrt_mcycle();
    
    // Setup the 1d loop with ssr (tell which streams to use, the size and the size of
    // the elements)
    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(double));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, chunk_per_core, sizeof(double));
    snrt_ssr_loop_1d(SNRT_SSR_DM2, chunk_per_core, sizeof(double));

    // Read from ft0 and ft1 that will be wired to a and b 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, src_a+offset); //ft0->a
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, src_b+offset); //ft1->b
    // Write to ft2 (sub)
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, dst+offset); //ft2->sub

    // Enable the SSRs
    snrt_ssr_enable();
    
    // Assembly code to sub ft0 and ft1 to ft2
    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fsub.d ft2, ft0, ft1\n"
        :
        : [ n_frep ] "r"(chunk_per_core - 1)
        : "ft0", "ft1", "ft2", "memory");

    // Disable SSRs
    snrt_ssr_disable();
    // Fence for FPU syncronization
    snrt_fpu_fence();

    *end_cycle = snrt_mcycle();

    return;
}