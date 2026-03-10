void dwt_opt(uint32_t chunk_per_core, uint32_t offset,
             float *x, float *y_low, float *y_high,
             float *h, float *g) {

    float zero = 0.0;
    
    snrt_mcycle();
    
    asm volatile(
    "flw ft3, 0(%[zero])\n"
    "flw ft4, 0(%[zero])\n"
    :
    : [zero] "r"(&zero)
    : "ft3","ft4");

    for (uint32_t n = 0; n < chunk_per_core; n++) {

        uint32_t out = offset + n;
        uint32_t center = 2 * out;
        uint32_t taps = (center < FILTER_LEN) ? (center + 1) : FILTER_LEN;

        snrt_ssr_loop_1d(SNRT_SSR_DM0, taps, -sizeof(float));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, taps, sizeof(float));

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, x + center);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, h);

        snrt_ssr_enable();

        asm volatile(
            "frep.o %[r], 1, 0, 0\n"
            "fmadd.s ft3, ft0, ft1, ft3\n"
            :
            : [r] "r"(taps - 1)
            : "ft0","ft1","ft3","memory");
        
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, x + center);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, g);

        asm volatile(
            "frep.o %[r], 1, 0, 0\n"
            "fmadd.s ft4, ft0, ft1, ft4\n"
            :
            : [r] "r"(taps - 1)
            : "ft0","ft1","ft4","memory");

        asm volatile(
            "fsw ft3, 0(%[yl])\n"
            "fsw ft4, 0(%[yh])\n"
            "fsub.s ft3, ft3, ft3\n" // Reset
            "fsub.s ft4, ft4, ft4\n"
            :
            : [yl] "r"(y_low + out),
            [yh] "r"(y_high + out)
            : "memory");
        
        snrt_ssr_disable();
        snrt_fpu_fence();
    }

    
    snrt_mcycle();
     
    return;
}
