#include <math.h>
#include "snrt.h"

snrt_barrier_t barr;

void ch_decomp_opt(uint32_t core_idx, uint32_t ncores,
                   uint64_t *start_cycle, uint64_t *end_cycle,
                   float *src, float *dst, uint32_t dim){
    *start_cycle = snrt_mcycle();

    for (uint32_t m = 0; m < dim; m++) {

        /* === Diagonal element === */
        uint32_t left = m % ncores;
        uint32_t block = m / ncores;
        uint32_t start = core_idx * block + (core_idx < left ? core_idx : left);
        uint32_t end = start + block + (core_idx < left ? 1 : 0);

    
        local_sum[core_idx] = 0.0f;
        for (uint32_t n = start; n < end; n++) {
            float x = dst[m*dim + n];
            local_sum[core_idx] += x * x;
        }
    

       snrt_partial_barrier(&barr, 8);

        if (core_idx == 0) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < ncores; c++)
                sum += local_sum[c];
            dst[m*dim + m] = sqrtf(dst[m*dim + m] - sum);
        }

        snrt_partial_barrier(&barr, 8);
     
        
        /* === Column update === */
        float lmm = dst[m*dim + m];
        float lmm_inv = 1.0f / (float) lmm;

        uint32_t col_left = (dim - (m + 1)) % ncores;
        uint32_t col_block = (dim - (m + 1)) / ncores;
        uint32_t col_start = core_idx * col_block + (core_idx < col_left ? core_idx : col_left) + (m + 1);
        uint32_t col_end = col_start + col_block + (core_idx < col_left ? 1 : 0);

        for (uint32_t n = col_start; n < col_end; n++) {
            float sum1 = 0.0;
            asm volatile (
                "flw ft3, 0(%0)" 
                :
                : "r"(&sum1) 
                : "ft3");
            // Avoid doing frep 64K times, maybe it is good!
            if(m>0){
                snrt_ssr_loop_1d(SNRT_SSR_DM0, m, sizeof(float));
                snrt_ssr_loop_1d(SNRT_SSR_DM1, m, sizeof(float));

                snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, dst + n*dim);
                snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, dst + m*dim);

                snrt_ssr_enable();

                asm volatile(
                    "frep.o %[n_frep], 1, 0, 0\n"
                    "fmadd.s ft3, ft0, ft1, ft3\n"
                    :
                    : [n_frep] "r"(m - 1)
                    : "ft0", "ft1", "ft3", "memory"
                );

                snrt_ssr_disable();
                snrt_fpu_fence();

                asm volatile(
                    "fsw ft3, 0(%0)" 
                    :
                    : "r"(&sum1) 
                    : "memory");

                
            }
            dst[n*dim + m] = (float) (src[n*dim + m] - sum1) * (float) lmm_inv;
        }

        

        snrt_partial_barrier(&barr, 8);
    }

    *end_cycle = snrt_mcycle();
}
