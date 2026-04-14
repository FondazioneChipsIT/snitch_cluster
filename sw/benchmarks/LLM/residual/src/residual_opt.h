// Luca Colombo Chips-IT 2025
/* Pretty simple, adds two matrices*/

void residual_opt(uint32_t core_idx, uint32_t chunk_per_core, uint32_t offset,
    float *x, float *y, float *out){

    // Chunk per core is the number of rows, we need the total amount
    uint32_t tot_ops = chunk_per_core*N; // number of elements to process per core

    // Setup the 1d loop with ssr (tell which streams to use, the size and the size of
    // the elements)
    snrt_ssr_loop_1d(SNRT_SSR_DM0, tot_ops, sizeof(float));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, tot_ops, sizeof(float));
    snrt_ssr_loop_1d(SNRT_SSR_DM2, tot_ops, sizeof(float));

    // Read from ft0 and ft1 that will be wired to a and b 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, x + offset*N); //ft0->x
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, y + offset*N); //ft1->y
    // Write to ft2 (out)
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out + offset*N); //ft2->out
   
    // Enable the SSRs
    snrt_ssr_enable();
    
    // Assembly code to add ft0 and ft1 to ft2
    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fadd.s ft2, ft0, ft1\n"
        :
        : [ n_frep ] "r"(tot_ops - 1)
        : "ft0", "ft1", "ft2", "memory");

    // Disable SSRs
    snrt_ssr_disable();
    // Fence for FPU syncronization
    snrt_fpu_fence();
    
    return;
}
