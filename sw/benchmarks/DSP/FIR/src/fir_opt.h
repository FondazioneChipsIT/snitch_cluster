// Luca Colombo Chips-IT 2025

void fir_opt(uint32_t chunk_per_core, uint32_t offset,
    double *x, double *y, double *h){

    double zero = 0.0;

    snrt_mcycle();

    for (uint32_t n = offset; n < offset + chunk_per_core; n++) {

        // Check if core would exceed its data boundaries
        uint32_t taps = (n < FILTER_LEN) ? (n + 1) : FILTER_LEN;
        // Read x backwards
        snrt_ssr_loop_1d(SNRT_SSR_DM0, taps, -sizeof(double));
        // Read h normally
        snrt_ssr_loop_1d(SNRT_SSR_DM1, taps, sizeof(double)); 
        
        snrt_ssr_enable();

        // Start from an offset n
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, x + n);
        // No offset for h
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, h);

        asm volatile(
        "fld ft3, 0(%[zero])\n"
        "frep.o %[n_frep], 1, 0, 0 \n"  /* Repeat elems times: ft3 = ft0 (mat_a) * ft1 (mat_b) + ft3 (acc)*/
        "fmadd.d ft3, ft0, ft1, ft3\n"
        "fsd ft3, 0(%[y_local])\n"
        :
        : [n_frep] "r"(taps - 1), [zero] "r"(&zero), [y_local] "r"(y+n)
        : "ft0", "ft1", "ft3", "memory");

        snrt_ssr_disable();
        snrt_fpu_fence();

    }
    
    snrt_mcycle();

    return;
}