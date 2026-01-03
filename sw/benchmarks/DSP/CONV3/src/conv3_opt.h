// Luca Colombo Chips-IT 2025
/* Conv3-> mutliply each time 3 times, so no need for C loops
   This helps us accelerate the kernel by using 2d loop*/


// Worse than the naive versione, 0.19 FLOP/cycle vs 0.30
void conv3_opt_V1(uint32_t chunk_per_core, uint32_t offset,
    double *x, double *y, double *h){

    double zero = 0.0;

    snrt_mcycle();


    // Read x 
    snrt_ssr_loop_1d(SNRT_SSR_DM0, CONV3_LEN, sizeof(double));
    // Read h 
    snrt_ssr_loop_1d(SNRT_SSR_DM1, CONV3_LEN, sizeof(double)); 

    for(uint32_t n=offset; n<chunk_per_core+offset;n++){

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
        : [n_frep] "r"(CONV3_LEN - 1), [zero] "r"(&zero), [y_local] "r"(y+n)
        : "ft0", "ft1", "ft3", "memory");

        snrt_ssr_disable();
        snrt_fpu_fence();
    }

    snrt_mcycle();

    return;
}

// Better than the naive versione, 0.40 FLOP/cycle vs 0.30!!!!

void conv3_opt_V2(uint32_t chunk_per_core, uint32_t offset,
    double *x, double *y, double *h){

    double zero = 0.0;

    snrt_mcycle();
    
    // Read x chunk per core times, each time 3 times, then go forward by 1 (after loop1, it goes back to the start, so need to +1)
    snrt_ssr_loop_2d(SNRT_SSR_DM0, CONV3_LEN, chunk_per_core, sizeof(double), sizeof(double));

    // Read h 3 times, then repeat for chunk_per_core_times
    snrt_ssr_loop_2d(SNRT_SSR_DM1, CONV3_LEN, chunk_per_core, sizeof(double) , 0); 

    // Write y
    snrt_ssr_loop_1d(SNRT_SSR_DM2, chunk_per_core, sizeof(double)); 

    snrt_ssr_enable();

    // Start from offset
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, x + offset);
    // No offset for h
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, h);
    // Y writeback
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, y + offset);

    asm volatile(  
    "frep.o %[n_frep], 6, 0, 0 \n" // Repeat chunk times
    "fld ft3, 0(%[zero])\n"        // acc = 0
    "fld ft4, 0(%[zero])\n"        // temp = 0
    "fmadd.d ft3, ft0, ft1, ft3\n" // 3 time like the filter size, need to change for CONV5 or 7
    "fmadd.d ft3, ft0, ft1, ft3\n"
    "fmadd.d ft3, ft0, ft1, ft3\n"
    "fadd.d ft2, ft3, ft4\n"     // storeback
    :
    : [n_frep] "r"(chunk_per_core-1),[zero] "r"(&zero)
    : "ft0", "ft1", "ft2", "ft3", "ft4", "memory");

    snrt_ssr_disable();
    snrt_fpu_fence();


    snrt_mcycle();

    return;
}
