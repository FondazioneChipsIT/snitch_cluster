#include "snrt.h"

void layernorm(float *input, float *output, uint32_t rows_core) {

    float zero = 0.0f;
    float embeddings_f = (float) EMBEDDINGS;

    asm volatile(
    "flw ft4, 0(%[zero])\n" // accumulator
    "flw ft7, 0(%[zero])\n" // accumulator for var
    "flw ft10, 0(%[embeddings_f])\n" // Embeddings
    "flw ft11, 0(%[EPS])\n"  // EPS
    :
    :[zero] "r"(&zero), [embeddings_f] "r" (&embeddings_f), [EPS] "r"(&EPS)
    :"ft3", "ft4", "ft7", "ft10", "ft11");

    snrt_ssr_enable();

    for(uint32_t curr_row = 0; curr_row < rows_core; curr_row ++){
        
        // Read each row 3 times
        snrt_ssr_loop_2d(SNRT_SSR_DM0, EMBEDDINGS, 3, sizeof(float), 0);
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, input + curr_row * EMBEDDINGS);
        //Output stream
        snrt_ssr_loop_1d(SNRT_SSR_DM1, EMBEDDINGS, sizeof(float));
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, output + curr_row * EMBEDDINGS);

        asm volatile(
            // First frep for mean value
            "frep.o %[n_frep], 1, 0, 0 \n" 
            "fadd.s ft4, ft0, ft4\n" // FT4 -> sum of EMBEDDINGS
            "fdiv.s ft5, ft4, ft10\n" // FT5 -> mean of EMBEDDINGS
            // Second frep for variance
            "frep.o %[n_frep], 2, 0, 0 \n" 
            "fsub.s ft6, ft0, ft5\n"
            "fmadd.s ft7, ft6, ft6, ft7\n" 
            "fdiv.s ft7, ft7, ft10\n" // FT7 -> var of EMBEDDINGS
            // Sqrt of var + eps
            "fadd.s ft7, ft7, ft11\n"
            "fsqrt.s ft7, ft7 \n" // FT7 -> sqrt(var + eps)
            // Final computation
            "frep.o %[n_frep], 2, 0, 0 \n"
            "fsub.s ft8, ft0, ft5 \n" // xi - mean
            "fdiv.s ft1, ft8, ft7 \n" // ft1 (output) = (xi-mean) / sqrt(var + eps)
            // Reset accumulator for next row
            "fsub.s ft4, ft4, ft4\n"
            "fsub.s ft7, ft7, ft7\n"
            :
            : [n_frep] "r"(EMBEDDINGS-1)
            : "ft0", "ft1", "ft4", "ft5", "ft6", "ft7", "ft8", "ft10", "ft11", "memory");

        snrt_fpu_fence();
    }

    snrt_ssr_disable();

    return;
}