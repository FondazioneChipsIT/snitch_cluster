
/* This code SHOULD perform better than the other, but instead does not*/


void dwt_opt(uint32_t chunk_per_core, uint32_t offset,
             double *x, double *y_low, double *y_high,
             double *h, double *g) {

    double zero = 0.0;
    
    snrt_mcycle();
    
    asm volatile(
    "fld ft3, 0(%[zero])\n"
    "fld ft4, 0(%[zero])\n"
    "fld ft5, 0(%[zero])\n"
    :
    : [zero] "r"(&zero)
    : "ft3","ft4", "ft5", "memory");

    snrt_ssr_loop_1d(SNRT_SSR_DM2, 1, sizeof(double));
    
    for (uint32_t n = 0; n < chunk_per_core; n++) {

        uint32_t out = offset + n;
        uint32_t center = 2 * out;
        uint32_t taps = (center < FILTER_LEN) ? (center + 1) : FILTER_LEN;

        // We read two times x
        snrt_ssr_loop_2d(SNRT_SSR_DM0, taps, 2, -sizeof(double), 0);
        snrt_ssr_loop_1d(SNRT_SSR_DM1, taps, sizeof(double));

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, x + center);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, h);
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, y_low + out);

        snrt_ssr_enable();

        asm volatile(
            "frep.o %[r], 1, 0, 0\n"
            "fmadd.d ft3, ft0, ft1, ft3\n"
            "fadd.d ft2, ft3, ft5\n"
            :
            : [r] "r"(taps - 1)
            : "ft0","ft1", "ft2", "ft3", "ft5", "memory");

        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, g);
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, y_high + out);
        
        asm volatile(
            "frep.o %[r], 1, 0, 0\n"
            "fmadd.d ft4, ft0, ft1, ft4\n"
            "fadd.d ft2, ft4, ft5\n"
            "fsub.d ft3, ft3, ft3\n" // Reset
            "fsub.d ft4, ft4, ft4\n" // Reset
            :
            : [r] "r"(taps - 1)
            : "ft0", "ft1", "ft3", "ft4", "ft5", "memory");

        snrt_ssr_disable();

        snrt_fpu_fence();
    }

    
    snrt_mcycle();
     
    return;
}
