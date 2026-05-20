#include "snrt.h"

void layernorm(float *input, float *output, float *weight, float *bias, uint32_t rows_core) {

    float zero = 0.0f;
    float embeddings_f = (float) EMBEDDINGS;

    asm volatile(
    "flw ft4, 0(%[zero])\n" // accumulator
    "flw ft5, 0(%[zero])\n" // accumulator
    "flw ft8, 0(%[zero])\n" // accumulator for var
    "flw ft9, 0(%[zero])\n" // accumulator for var
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

        snrt_ssr_loop_1d(SNRT_SSR_DM2, EMBEDDINGS, sizeof(float));
        snrt_ssr_read (SNRT_SSR_DM2, SNRT_SSR_1D, weight);

        asm volatile(
            // First frep for mean value
            "frep.o %[n_frep], 2, 0, 0 \n" 
            "fadd.s ft4, ft0, ft4\n" // FT4 -> sum of EMBEDDINGS
            "fadd.s ft5, ft0, ft5\n" // FT5 -> sum of EMBEDDINGS
            "fadd.s ft5, ft5, ft4\n" 
            "fdiv.s ft5, ft5, ft10\n" // FT5 -> mean of EMBEDDINGS
            // Second frep for variance
            "frep.o %[n_frep], 4, 0, 0 \n" 
            "fsub.s ft6, ft0, ft5\n"
            "fsub.s ft7, ft0, ft5\n"
            "fmadd.s ft8, ft6, ft6, ft8\n" 
            "fmadd.s ft9, ft7, ft7, ft9\n" 
            "fadd.s ft7, ft8, ft9\n" 
            "fdiv.s ft7, ft7, ft10\n" // FT7 -> var of EMBEDDINGS
            // Sqrt of var + eps
            "fadd.s ft7, ft7, ft11\n"
            "fsqrt.s ft7, ft7 \n" // FT7 -> sqrt(var + eps)
            // Final computation
            "frep.o %[n_frep], 6, 0, 0 \n"
            "fsub.s ft8, ft0, ft5 \n" // xi - mean
            "fsub.s ft9, ft0, ft5 \n" // xi - mean
            "fdiv.s ft10, ft8, ft7 \n" // ft10 = (xi-mean) / sqrt(var + eps)
            "fdiv.s ft11, ft9, ft7 \n" // ft11 = (xi-mean) / sqrt(var + eps)
            "fmul.s ft1, ft10, ft2\n"  // xnorm_i   * w_i   
            "fmul.s ft1, ft11, ft2\n"  // xnorm_i+1 * w_i+1 

            // Reset accumulators for next row
            "fsub.s ft4, ft4, ft4\n"
            "fsub.s ft5, ft5, ft5\n"
            "fsub.s ft8, ft8, ft8\n"
            "fsub.s ft9, ft9, ft9\n"
            :
            : [n_frep] "r"(EMBEDDINGS/2-1)
            : "ft0", "ft1", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "ft11", "memory");

        // The fence is CRITICAL to ensure that all the computations are done before the next iteration of the loop starts and reads 
        // the output
        snrt_fpu_fence();

        //  DM0=READ output, DM1=WRITE output, DM2=bias 
        // SSRs do not go in conflict as the read of DM0 and DM2 happens before the write of DM1
        snrt_ssr_loop_1d(SNRT_SSR_DM0, EMBEDDINGS, sizeof(float));
        snrt_ssr_read (SNRT_SSR_DM0, SNRT_SSR_1D, output + curr_row * EMBEDDINGS);
        // Dont need to reconfigure for DM1 and DM2 as they are already set for 1D access with the right stride
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, output + curr_row * EMBEDDINGS);
        snrt_ssr_read (SNRT_SSR_DM2, SNRT_SSR_1D, bias);

        // Loop 4
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0\n"
            "fadd.s ft1, ft0, ft2\n"           // xnorm*w + b_i  
            :
            : [n_frep] "r"(EMBEDDINGS - 1)
            : "ft0", "ft1", "ft2", "memory"
        );

        snrt_fpu_fence();
    }

    snrt_ssr_disable();
    snrt_fpu_fence();
    return;
}