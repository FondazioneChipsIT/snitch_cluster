#include <math.h>
#include "snrt.h"

snrt_barrier_t barr;

void cholesky_opt(uint32_t core_idx, uint32_t ncores,
                   float *src, float *dst, uint32_t dim){

    snrt_mcycle();
    for (uint32_t m = 0; m < dim; m++) {

        // Only core 0 computes the diagonal element
        if (core_idx == 0) {
            float sum = 0.0f;

            // Use ssrs only when m is big enough
            if( m>7 ){
                
                snrt_ssr_loop_1d(SNRT_SSR_DM0, m, sizeof(float));
                snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, dst + m*dim);

                snrt_ssr_enable();

                asm volatile(
                "fmv.s.x ft3,x0\n"
                "frep.o %[n_frep], 1, 0, 0 \n" 
                "fmadd.s ft3, ft0, ft0, ft3\n" 
                "fsw ft3, 0(%[sum])\n" //storeback
                :
                : [n_frep] "r"(m - 1), [sum] "r"(&sum)
                : "ft0", "ft3" "memory");

                snrt_ssr_disable();
                snrt_fpu_fence();

            }else{
                for (uint32_t k = 0; k < m; k++) {
                    float v = dst[m*dim + k];
                    sum += v * v;
                }
            }

            float diag = src[m*dim + m] - sum;

            // SAFE GUARD, avoid possible Nan/inf 
            if (diag <= 0.0f) {
                diag = 1e-12f;
            }

            dst[m*dim + m] = sqrtf(diag);
        }

        // Other cores wait for the diagonal element to be computed before proceeding
        snrt_partial_barrier(&barr, 8);

        float lmm = dst[m*dim + m];
        float inv_lmm = 1.0f / lmm;

        // Divide the work
        uint32_t total_elems = dim - (m + 1);
        uint32_t chunk_per_core = total_elems / ncores;
        uint32_t rem   = total_elems % ncores;

        // We start from m+1 because the first m elements of the column are already computed
        // At the end of the diag computation (core zero if)
        uint32_t start = (m + 1) + core_idx * chunk_per_core + (core_idx < rem ? core_idx : rem);
        // Split the remaining elements among first 7 cores, the eight core will never get another elment
        uint32_t end   = start + chunk_per_core + (core_idx < rem ? 1 : 0);

        for (uint32_t n = start; n < end; n++) {

            float sum = 0.0f;

            // dot product corretto
            if( m > 7){
                snrt_ssr_loop_1d(SNRT_SSR_DM0, m, sizeof(float));
                snrt_ssr_loop_1d(SNRT_SSR_DM1, m, sizeof(float));

                snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, dst + n*dim);
                snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, dst + m*dim);

                snrt_ssr_enable();

                asm volatile(
                    "fmv.s.x ft3,x0\n"
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
                    : "r"(&sum) 
                    : "memory");


            }else{
                for (uint32_t k = 0; k < m; k++) {
                    sum += dst[n*dim + k] * dst[m*dim + k];
                }
            }
            
            float val = src[n*dim + m] - sum;
            dst[n*dim + m] = val * inv_lmm;
        }

        // Barrier to avoid that core 0 starts to compute the next diagonal element before all the elements of the current column are computed
        snrt_partial_barrier(&barr, 8);
    }

    snrt_mcycle();
    return;
}