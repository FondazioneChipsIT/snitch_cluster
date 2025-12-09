// Luca Colombo Chips-IT 2025
/* Each core searches in its chunk the minimum LOCAL value, then core 0 reads the ncore results in main
   to search the absolute minimum value. We firstly load in ft3 the first element each time to have
   something to compare to */

void vect_min_opt(uint32_t core_idx, uint32_t chunk_per_core, uint32_t offset, 
    uint64_t *start_cycle, double *src, double *min_core){

    *start_cycle = snrt_mcycle();

    // Setup the 1d loop with ssr (tell which streams to use, the size and the size of
    // the elements)
    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(double));

    // Read from ft0 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, src + offset); //ft0<-src

    // Loads first value of vector in ft3
    asm volatile(
        "fld ft3, 0(%[base])\n"    // ft3 = src[offset]
        :
        : [base] "r"(src + offset)
        : "ft3", "memory");

    // Enable the SSRs
    snrt_ssr_enable();
    
    // Assembly code to compare value of vector to one stored in ft3 and change ft3 if it is minimum
    asm volatile(
        "frep.o %[n_frep], 1, 0, 0 \n"
        "fmin.d ft3, ft3, ft0\n"
        :
        : [ n_frep ] "r"(chunk_per_core - 1)
        : "ft0", "ft3", "memory");

    // Disable SSRs
    snrt_ssr_disable();

    // Save the local core results in shared memory
    asm volatile("fsd ft3, (%[res])" :: [res] "r"(min_core + core_idx) : "memory");
    
    snrt_fpu_fence(); // Syncronize 

    return;
}