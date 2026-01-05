// Luca Colombo Chips-IT 2025
/* Conv3x3 matrix, using 4d loops to read 3 elements of x from 3 rows and
    then move to next element in current row*/

void conv3x3_opt(uint32_t core_idx, uint32_t chunk_per_core, uint32_t offset,
    double *x, double *y, double *h){

    double zero = 0.0; 
    
    snrt_mcycle();
    // Check if last two cores, for 6 need 2 rows at least and 7 3
    if(core_idx < 6 || (chunk_per_core > 1 && core_idx == 6) || (chunk_per_core > 2 && core_idx == 7)){

        uint32_t tot_ops = (LEN-2)*chunk_per_core;
        if(core_idx == 7){
            tot_ops = (LEN-2)*(chunk_per_core-2);
        }
        // initialize acc1 and acc2 to 0
        asm volatile(
            "fld ft3, 0(%[zero])\n"
            "fld ft4, 0(%[zero])\n"
            :
            : [zero] "r"(&zero)
            : "ft3", "ft4", "memory");

        // Read 3 elements of x (stride 1), move to the next row (stride LEN),
        // repeat this process 3 time (9 elements). Then read all elements of 
        // the row minus the last two (no zero padding), then move to the next 
        // row (stride LEN)
        snrt_ssr_loop_4d(SNRT_SSR_DM0, CONV3x3_LEN, CONV3x3_LEN, LEN-2, 
                        chunk_per_core, sizeof(double), 
                        LEN*sizeof(double), sizeof(double), 
                        LEN*sizeof(double));

        // Read h 9 times, then repeat for tot_ops
        snrt_ssr_loop_2d(SNRT_SSR_DM1, CONV3x3_LEN*CONV3x3_LEN, tot_ops, sizeof(double), 0); 

        // Write y, only tot_ops times (num of rows* LEN-2), 
        // do not use padding (avoid reading the last two elements)
        snrt_ssr_loop_1d(SNRT_SSR_DM2, tot_ops, sizeof(double)); 

        snrt_ssr_enable();

        // Start from offset
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, x + offset*LEN);
        // No offset for h
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, h);
        // Y writeback
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, y + offset*(LEN-2));

        asm volatile(  
        "frep.o %[n_frep], 12, 0, 0 \n" // Repeat chunk times
        "fsub.d ft3, ft3, ft3\n"   // acc = 0
        "fsub.d ft4, ft4, ft4 \n"  // acc2= 0
        "fmadd.d ft3, ft0, ft1, ft3\n" // 9 time like the filter size, need to change for CONV5 or 7
        "fmadd.d ft4, ft0, ft1, ft4\n"
        "fmadd.d ft3, ft0, ft1, ft3\n"
        "fmadd.d ft4, ft0, ft1, ft4\n"
        "fmadd.d ft3, ft0, ft1, ft3\n"
        "fmadd.d ft4, ft0, ft1, ft4\n"
        "fmadd.d ft3, ft0, ft1, ft3\n"
        "fmadd.d ft4, ft0, ft1, ft4\n"
        "fmadd.d ft3, ft0, ft1, ft3\n" 
        "fadd.d ft2, ft3, ft4\n"     // storeback
        :
        : [n_frep] "r"(tot_ops-1)
        : "ft0", "ft1", "ft2", "ft3", "ft4", "memory");

        snrt_ssr_disable();
        snrt_fpu_fence();


        snrt_mcycle();
    }
    return;
}
