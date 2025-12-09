// Luca Colombo Chips-IT 2025

void vect_scale_opt(uint32_t chunk_per_core, uint32_t offset, uint64_t *start_cycle,uint64_t *end_cycle, 
    double *src, double *dst, double *val){

    *start_cycle = snrt_mcycle();

    // Setup the 1d loop with ssr (tell which streams to use, the size and the size of
    // the elements)
    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(double));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, chunk_per_core, sizeof(double));

    // Read from ft0 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, src + offset); //ft0<-src
    // Write to ft1
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, dst + offset); //ft1->dst
    
    // Load the scalar value val in ft3 register
    asm volatile(
        "fld ft3, 0(%[val])\n"
        :                   // Outputs
        : [val] "r"(val)    // Inputs 
        : "ft3");           // Clobber list

    // Enable the SSRs
    snrt_ssr_enable();
    
    // Assembly code to multiply ft0 to ft3 and store in ft1
    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fmul.d ft1, ft0, ft3\n"
        :
        : [ n_frep ] "r"(chunk_per_core - 1)
        : "ft0", "ft1", "ft3", "memory");

    // Disable SSRs
    snrt_ssr_disable();
    // Fence for FPU syncronization
    snrt_fpu_fence();

    *end_cycle = snrt_mcycle();

    return;
}