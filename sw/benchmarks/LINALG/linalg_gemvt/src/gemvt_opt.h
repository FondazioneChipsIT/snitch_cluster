// Luca Colombo Chips-IT 2026
void gemvt_opt(uint32_t chunk_per_core, uint32_t offset,
                    float *mat_a, float* vec_x, float alpha, float *dst, uint32_t m , uint32_t n) {

    float zero = 0.0f; // Zero register

    // Columns are assigned to the cores INTERLEAVED (core c takes the columns
    // c, c+ncores, c+2*ncores, ...) instead of in one contiguous block: a
    // column of mat_a is read with stride n*sizeof(float) = 512 B, and with 32
    // TCDM banks of 4 B the whole column lives in the single bank (col % 32).
    // With the contiguous mapping the cores 0-2, 1-3, 4-6 and 5-7 sit on the
    // same bank for the whole kernel and their accesses are serialized.
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores   = snrt_cluster_compute_core_num();

    // Read element of column, then go to next row.
    // then change column (ncores columns ahead), repeat chunk_per_core times
    snrt_ssr_loop_2d(SNRT_SSR_DM0, m, chunk_per_core, n*sizeof(float), ncores*sizeof(float));
    // Read the vec_x vector with a 1 stride-> access an element, finish the vector, repeat chunk_per_core times
    snrt_ssr_loop_2d(SNRT_SSR_DM1, m, chunk_per_core, sizeof(float), 0);

    snrt_ssr_loop_1d(SNRT_SSR_DM2, chunk_per_core, ncores*sizeof(float));
    
    /* Load zero into ft3, used as accumulator */
    asm volatile(
    "flw ft3, 0(%[zero])\n"
    "flw ft4, 0(%[zero])\n"
    "flw ft5, 0(%[zero])\n"
    "flw ft6, 0(%[zero])\n"
    "flw ft7, 0(%[alpha])\n"
    :: [zero] "r"(&zero), [alpha] "r"(&alpha)
    : "ft3", "ft4", "ft5", "ft6", "ft7");
    snrt_ssr_enable();

    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, mat_a + core_idx); // First column of this core, stride n to walk it, then ncores columns ahead
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, vec_x);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, dst + core_idx); // Write the interleaved results to dst
        
    for(uint32_t i = 0; i < chunk_per_core; i++){
        asm volatile(
           "frep.o %[n_frep], 4, 0, 0 \n"  
            "fmadd.s ft3, ft0, ft1, ft3\n" // A*x
            "fmadd.s ft4, ft0, ft1, ft4\n" // A*x
            "fmadd.s ft5, ft0, ft1, ft5\n" // A*x
            "fmadd.s ft6, ft0, ft1, ft6\n" // A*x
            "fadd.s ft3, ft3, ft4\n" // A*x
            "fadd.s ft5, ft5, ft6\n" // A*x
            "fadd.s ft3, ft3, ft5\n" // A*x
            "fmul.s ft2, ft3, ft7\n" // alpha*(A*x), also store back in memory
            "fsub.s ft3, ft3, ft3\n"
            "fsub.s ft4, ft4, ft4\n"
            "fsub.s ft5, ft5, ft5\n"
            "fsub.s ft6, ft6, ft6\n"
            :
            : [n_frep] "r"(m/4 - 1)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "memory");
    }

    snrt_ssr_disable();
    snrt_fpu_fence();
    return;
}