// Luca Colombo Chips-IT 2026
void gemv_opt(uint32_t chunk_per_core, uint32_t offset,
                    float *mat_a, float* vec_x, float alpha, float *dst, uint32_t n) {

    float zero = 0.0f; // Zero register
    // Read the mat_a matrix with a 1 stride-> access a row
    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core*n, sizeof(float));

    // Read the vec_x vector with a 1 stride-> access an element, finish the vector, repeat chunk_per_core times
    snrt_ssr_loop_2d(SNRT_SSR_DM1, n, chunk_per_core, sizeof(float), 0);

    snrt_ssr_loop_1d(SNRT_SSR_DM2, chunk_per_core, sizeof(float)); 
    
    /* Load zero into ft3, used as accumulator */
    asm volatile(
    "flw ft3, 0(%[zero])\n"
    "flw ft4, 0(%[alpha])\n"
    :: [zero] "r"(&zero), [alpha] "r"(&alpha)
    : "ft3", "ft4");

    snrt_ssr_enable();

        
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat_a + offset * n); // Read the chunk of mat_a for this core
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, vec_x);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, dst + offset); // Write the result chunk to dst
        
    for(uint32_t i = 0; i < chunk_per_core; i++){
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"  
            "fmadd.s ft3, ft0, ft1, ft3\n" // A*x
            "fmul.s ft2, ft3, ft4\n" // alpha*(A*x)
            "fsub.s ft3, ft3, ft3\n"
            :
            : [n_frep] "r"(n - 1)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "memory");
    }

    snrt_ssr_disable();
    snrt_fpu_fence();
    return;
}