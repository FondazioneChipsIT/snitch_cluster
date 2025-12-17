void dwt_opt(uint32_t chunk_per_core, uint32_t offset,
             double *x, double *y_low, double *y_high,
             double *h, double *g) {

    double zero = 0.0;
    
    snrt_mcycle();

    
   
    for (uint32_t n = offset; n < offset+chunk_per_core; n++) {

        uint32_t taps = (2*n < FILTER_LEN) ? (2*n + 1) : FILTER_LEN;

        asm volatile(
            "fld ft3, 0(%[zero])\n"
            "fld ft4, 0(%[zero])\n"
            :
            : [zero] "r"(&zero)
            : "ft3","ft4","memory");

        uint32_t idx = 2*(offset + n);  // downsample by 2

        snrt_ssr_loop_1d(SNRT_SSR_DM0, taps, -sizeof(double));  // x stride -1
        snrt_ssr_loop_1d(SNRT_SSR_DM1, taps, sizeof(double));   // h stride +1

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, x + idx);  // read x
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, h);        // read low-pass

        snrt_ssr_enable();

        // accum low-pass
        asm volatile(
            "frep.o %[taps], 1, 0, 0\n"
            "fmadd.d ft3, ft0, ft1, ft3\n"
            "fsd ft3, 0(%[y_low])\n"
            :
            : [zero] "r"(&zero), [taps] "r"(taps), [y_low] "r"(y_low + n)
            : "ft0","ft1","ft3","memory");

        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, g);        // read high-pass
        // accum high-pass
        asm volatile(
            "frep.o %[taps], 1, 0, 0\n"
            "fmadd.d ft4, ft0, ft1, ft4\n"
            "fsd ft4, 0(%[y_high])\n"
            :
            : [zero] "r"(&zero), [taps] "r"(taps), [y_high] "r"(y_high + n)
            : "ft0","ft1","ft3","memory");

        snrt_ssr_disable();
        snrt_fpu_fence();

    }
    snrt_mcycle();
     
    return;
}
