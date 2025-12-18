// Luca Colombo Chips-IT 2025
/* Conv3-> mutliply each time 3 times, so no need for C loops
   This helps us accelerate the kernel by using 2d loop*/
void conv3_opt(uint32_t chunk_per_core, uint32_t offset,
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
// Could work, need to fix, is it worth the effort?

/* uint32_t tot_ops = chunk_per_core*CONV3_LEN;
    
    // Read x chunk per core times, each time 3 times, then roll back by 1
    snrt_ssr_loop_2d(SNRT_SSR_DM0,chunk_per_core, CONV3_LEN,-(CONV3_LEN-2)*sizeof(double),sizeof(double));

    // Read h 3 times, then rollback for chunk_per_core_times
    snrt_ssr_loop_2d(SNRT_SSR_DM1, chunk_per_core, CONV3_LEN, -(CONV3_LEN-1)*sizeof(double), sizeof(double)); 

    // Write y
    snrt_ssr_loop_1d(SNRT_SSR_DM2, chunk_per_core, sizeof(double)); 

    snrt_ssr_enable();

    // Start from an offset n
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, x + offset);
    // No offset for h
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, h);
    // Y writeback
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, y+offset);

    asm volatile(  
    "frep.o %[n_frep], 6, 0, 0 \n" // Repeat tot_ops times
    "fld ft3, 0(%[zero])\n"        // acc = 0
    "fld ft4, 0(%[zero])\n"        // temp = 0
    "fmadd.d ft3, ft0, ft1, ft3\n" // 3 time like the filter size
    "fmadd.d ft3, ft0, ft1, ft3\n"
    "fmadd.d ft3, ft0, ft1, ft3\n"
    "fadd.d ft2, ft3, ft4\n"     // storeback
    :
    : [n_frep] "r"(tot_ops-1),[zero] "r"(&zero)
    : "ft0", "ft1", "ft2", "ft3", "ft4", "memory");*/
