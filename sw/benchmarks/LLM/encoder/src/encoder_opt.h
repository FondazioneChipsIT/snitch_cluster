// Luca Colombo Chips-IT 2026

void encoder_opt( uint32_t chunk_per_core, uint32_t offset, uint32_t bb, uint32_t tt,
    int *inp, float *wte_TCDM, float *wpe_TCDM, float *out_TCDM){

    for (uint32_t i = 0; i < bb * tt; i++) {
        // We dont need B as we have split the work among the cores
        uint32_t t = i % tt;
        uint32_t ix = inp[i];          

        // Read wte at the right index
        snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core, sizeof(float));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, chunk_per_core, sizeof(float));
        snrt_ssr_loop_1d(SNRT_SSR_DM2, chunk_per_core, sizeof(float));
        
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, wte_TCDM + ix * C + offset);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, wpe_TCDM + t * C + offset);
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out_TCDM + i * C + offset);

        snrt_ssr_enable();

        asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"  // Repeat C times: ft0 = wte (wte) + ft1 (wpe)
                "fadd.s ft2, ft0, ft1\n"  // out = wte + wpe
                :
                : [n_frep] "r"(chunk_per_core - 1)
                : "ft0", "ft1", "ft2", "memory");
            
        snrt_ssr_disable();   

    }

    snrt_fpu_fence();

    return;
}
