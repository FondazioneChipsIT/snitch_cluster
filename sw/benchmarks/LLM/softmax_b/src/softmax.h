// 2026 Luca Colombo Chips-IT
// Taken from the kernels folder, written by the people there reported!

#include "snrt.h"

#include "fastexpf.h"
float max_core[8], sum[8];
float core0max, global_sum;
snrt_barrier_t   barr;

void softmax_FP32(float *input, float *output, uint32_t batch_size, uint32_t seq_len, uint32_t input_samples) {

    uint32_t core_idx = snrt_cluster_core_idx();
    float zero = 0.0f;
    asm volatile(
        "flw ft4, 0(%[zero])\n"
        :
        : [zero] "r"(&zero)
        :  "ft4", "memory");

    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {

            max_core[core_idx] = input[b * seq_len * input_samples + s * input_samples];
            sum[core_idx] = 0.0f;

            for (uint32_t i = 1; i < input_samples; i++) {
                if (input[b * seq_len * input_samples + s * input_samples + i] > max_core[core_idx]) {
                    max_core[core_idx] = input[b * seq_len * input_samples + s * input_samples + i];
                }
            }

            snrt_partial_barrier(&barr, 8);

            if (core_idx == 0) {
                core0max = max_core[0];
                for (uint32_t i = 1; i < 8; i++) {
                    if (max_core[i] > core0max) {
                        core0max = max_core[i];
                    }
                }
            }

            snrt_partial_barrier(&barr, 8);

            for (uint32_t i = 0; i < input_samples; i++) {
                output[b * seq_len * input_samples + s * input_samples + i] = 
                    fast_expf(input[b * seq_len * input_samples + s * input_samples + i] - core0max);

                snrt_ssr_loop_1d(SNRT_SSR_DM0, input_samples, sizeof(float)); 
                snrt_ssr_loop_1d(SNRT_SSR_DM1, 1, sizeof(float)); 

                snrt_ssr_enable();

                snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, output + b * seq_len * input_samples + s * input_samples);
                snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, sum + core_idx);

                asm volatile(  
                "fsub.s ft3, ft3, ft3\n"   // acc = 0
                "frep.o %[n_frep], 1, 0, 0 \n" // Repeat chunk times
                "fadd.s ft3, ft3, ft0\n" 
                "fadd.s ft2, ft3, ft4\n" // Storeback
                :
                : [n_frep] "r"(input_samples-1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "memory");

                snrt_ssr_disable();

                // same as 
                // sum[core_idx] += output[b * seq_len * input_samples + s * input_samples + i];
            }

            snrt_partial_barrier(&barr, 8);

            if (core_idx == 0) {
                float gsum = 0.0f;
                for (uint32_t c = 0; c < 8; c++)
                    gsum += sum[c];
                global_sum = gsum;
            }

            snrt_partial_barrier(&barr, 8);

            for (uint32_t i = 0; i < input_samples; i++) {
                output[b * seq_len * input_samples + s * input_samples + i] /= global_sum;
            }

            snrt_partial_barrier(&barr, 8);
        }
    }

    return;
}