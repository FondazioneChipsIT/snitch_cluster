// Luca Colombo Chips-IT 2025

void matrix_swap_rows_opt2(uint32_t chunk_per_core, uint32_t offset, 
                        float *mat_a, float *row_a, float *row_b,
                        uint32_t indx_a, uint32_t indx_b){

    float zero = 0.0f;

    // Setup the 1d loop with ssr (tell which streams to use, the size and the size of
    // the elements)
    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(float));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, chunk_per_core, sizeof(float));
    // Write to ft0
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat_a + indx_a*elems + offset); //ft0<-mat_a
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, row_a + offset); //ft1->row_a

    // Load zero in ft4
    asm volatile(
        "flw ft4, 0(%[zero])\n"
        :                   // Outputs
        : [zero] "r"(zero)   // Inputs 
        : "ft4");           // Clobber list
    
    snrt_ssr_enable();
    // Assembly code to add ft3 (val) to ft4 (0) and store in ft1 (vec)
    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fadd.s ft1, ft0, ft4\n"
        :
        : [ n_frep ] "r"(chunk_per_core - 1)
        : "ft0", "ft3", "ft4", "memory");

    // Disable SSRs
    snrt_ssr_disable();

    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(float));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, chunk_per_core, sizeof(float));
    // Write to ft0
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat_a + indx_b*elems + offset); //ft0<-mat_a
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, row_b + offset); //ft1->row_a
    
    snrt_ssr_enable();
     // Assembly code to add ft3 (val) to ft4 (0) and store in ft1 (vec)
    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fadd.s ft1, ft0, ft4\n"
        :
        : [ n_frep ] "r"(chunk_per_core - 1)
        : "ft0", "ft3", "ft4", "memory");

    // Disable SSRs
    snrt_ssr_disable();


    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(float));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, chunk_per_core, sizeof(float));
    // Write to ft0
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, row_a + offset); //ft0<-row_a
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, mat_a + indx_b*elems + offset); //ft1->mat_a
    
    snrt_ssr_enable();
     // Assembly code to add ft3 (val) to ft4 (0) and store in ft1 (vec)
    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fadd.s ft1, ft0, ft4\n"
        :
        : [ n_frep ] "r"(chunk_per_core - 1)
        : "ft0", "ft3", "ft4", "memory");

    // Disable SSRs
    snrt_ssr_disable();


    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(float));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, chunk_per_core, sizeof(float));
    // Write to ft0
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, row_b + offset); //ft0<-row_b
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, mat_a + indx_a*elems + offset); //ft1->mat_a
    
    snrt_ssr_enable();
     // Assembly code to add ft3 (val) to ft4 (0) and store in ft1 (vec)
    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fadd.s ft1, ft0, ft4\n"
        :
        : [ n_frep ] "r"(chunk_per_core - 1)
        : "ft0", "ft3", "ft4", "memory");

    // Disable SSRs
    snrt_ssr_disable();


    // Fence for FPU syncronization
    snrt_fpu_fence();

    return;
}