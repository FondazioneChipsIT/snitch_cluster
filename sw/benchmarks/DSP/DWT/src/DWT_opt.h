void dwt_opt(uint32_t chunk_per_core, uint32_t offset,
             double *x, double *y_low, double *y_high,
             double *h, double *g) {

    double zero = 0.0;
    
    snrt_mcycle();

    for (uint32_t n = 0; n < chunk_per_core; n++) {

        uint32_t out = offset + n;
        uint32_t center = 2 * out;
        uint32_t taps = (center < FILTER_LEN) ? (center + 1) : FILTER_LEN;
        uint32_t reps = taps - 1;

        asm volatile(
            "fld ft3, 0(%[zero])\n"
            "fld ft4, 0(%[zero])\n"
            :
            : [zero] "r"(&zero)
            : "ft3","ft4");

        snrt_ssr_loop_1d(SNRT_SSR_DM0, taps, -sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, taps, sizeof(double));

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, x + center);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, h);

        snrt_ssr_enable();

        asm volatile(
            "frep.o %[r], 1, 0, 0\n"
            "fmadd.d ft3, ft0, ft1, ft3\n"
            :
            : [r] "r"(reps)
            : "ft0","ft1","ft3","memory");
        
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, x + center);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, g);

        asm volatile(
            "frep.o %[r], 1, 0, 0\n"
            "fmadd.d ft4, ft0, ft1, ft4\n"
            :
            : [r] "r"(reps)
            : "ft0","ft1","ft4","memory");

        asm volatile(
            "fsd ft3, 0(%[yl])\n"
            "fsd ft4, 0(%[yh])\n"
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
