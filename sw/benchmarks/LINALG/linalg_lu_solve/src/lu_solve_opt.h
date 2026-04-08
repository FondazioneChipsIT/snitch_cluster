// Luca Colombo Chips-IT 2025
// Optimized version with SSR and FREP

/*
# Lu solve 64x64 performance
# Mean cycles: 75596
# Mean FLOP/cycle: 0.107518
# Total FLOP/cycle: 0.860141

# Lu solve 32x32 performance
# Mean cycles: 36851
# Mean FLOP/cycle: 0.054706
# Total FLOP/cycle: 0.437645

OPT? more like better not to use ssrs in this case

*/

snrt_barrier_t barr2;
void lu_solve_opt(uint32_t core_idx, uint32_t ncores, uint64_t *offset_cycle, uint64_t *end_cycle, 
                float *mat, uint32_t *perm, float *y, float *vec, float *result) {

    float zero = 0.0f;
    uint32_t perm_idx;

    // Start cycle count
    *offset_cycle = snrt_mcycle();
    
    // -------------------------
    // FORWARD SUBSTITUTION (L * y = P * vec)
    // -------------------------
    for (uint32_t m = 0; m < elems; m++) {

            /* Load zero into ft3 */
        asm volatile(
        "flw ft3, 0(%[zero])\n"
        :
        : [zero] "r"(&zero)
        : "ft3");
    

        int block = m / (int)ncores;
        int left  = m % (int)ncores;

        // offset/end per questo core (distribuzione dei resti sui primi 'left' core)
        int offset = core_idx * block + (core_idx < (uint32_t)left ? core_idx : left);
        int end   = offset + block + (core_idx < (uint32_t)left ? 1 : 0);

        int chunk_per_core = end - offset;

        if(chunk_per_core>0){ // avoid ssr and frep erros
            // mat and y
            snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(float));
            snrt_ssr_loop_1d(SNRT_SSR_DM1, chunk_per_core, sizeof(float));


            // read the matrix and y
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat + m * elems + offset); 
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, y + offset);
            
            
            snrt_ssr_enable();

            asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"  // repeat the next 1 instruction
            "fmadd.s ft3, ft0, ft1, ft3\n" // ft3 = ft0*ft1 +ft3
            :
            : [n_frep] "r"(chunk_per_core - 1)
            : "ft0", "ft1",  "ft3", "memory");
            
            snrt_ssr_disable();
            snrt_fpu_fence();
        }
        // Store back in local_sum
            asm volatile(
            "fsw ft3, 0(%[local_sum])\n"
            :
            : [local_sum] "r"(&local_sum[core_idx])
            : "ft3");
           
        


        snrt_partial_barrier(&barr2, 8);

        // Core 0 sums, not parallelizable
        if (core_idx == 0) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < ncores; i++)
                sum += local_sum[i];                          

            perm_idx = perm[m];
            y[m] = vec[perm_idx] - sum;
        }
        
        // Wait for new ys
        snrt_partial_barrier(&barr2, 8);
    }

    snrt_partial_barrier(&barr2, 8);


    // BACKWARD SUBSTITUTION (U * result = y)
    
    for (int m = (int) elems - 1; m >=0; m--) {
        


            /* Load zero into ft3 */
        asm volatile(
        "flw ft3, 0(%[zero])\n"
        :
        : [zero] "r"(&zero)
        : "ft3");
    
        
        int count = (int)elems - (m + 1);
        int block = count / (int)ncores;
        int left  = count % (int)ncores;

        int offset_rel = core_idx * block + (core_idx < (uint32_t)left ? core_idx : left);
        int end_rel   = offset_rel + block + (core_idx < (uint32_t)left ? 1 : 0);

        int offset = (m + 1) + offset_rel;
        int end   = (m + 1) + end_rel;

        int chunk_per_core = end - offset;
        
        if(chunk_per_core>0){ // avoid ssr and frep erros
            // mat and y
            snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(float));
            snrt_ssr_loop_1d(SNRT_SSR_DM1, chunk_per_core, sizeof(float));


            // read the matrix and result
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, mat + m * elems + offset); 
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, result + offset); 
            
            
            snrt_ssr_enable();

            asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"  // repeat the next 1 instruction
            "fmadd.s ft3, ft0, ft1, ft3\n" // ft3 = ft0*ft1 +ft3
            :
            : [n_frep] "r"(chunk_per_core- 1)
            : "ft0", "ft1",  "ft3", "memory");
            
            snrt_ssr_disable();
            snrt_fpu_fence();

        }

        // Store back in local_sum
            asm volatile(
            "fsw ft3, 0(%[local_sum])\n"
            :
            : [local_sum] "r"(&local_sum[core_idx])
            : "ft3");
        

        snrt_partial_barrier(&barr2, 8);

        // Core 0 sums
        if (core_idx == 0) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < ncores; i++)
                sum += local_sum[i];
            result[m] = (y[m] - sum) / mat[m * elems + m];
        }
     
        snrt_partial_barrier(&barr2, 8);
    }

    *end_cycle = snrt_mcycle();
}

    


