// 2026 Luca Colombo Chips-IT
// Taken from the kernels folder, written by the people there reported!

#include "snrt.h"

#include "fastexpf.h"
float max_core[8], sum[8];
float core0max, global_sum;
snrt_barrier_t   barr;

void softmax_FP32(float *input, float *output, uint32_t batch_size, uint32_t seq_len, uint32_t input_samples) {

    uint32_t core_idx = snrt_cluster_core_idx();

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
                sum[core_idx] += output[b * seq_len * input_samples + s * input_samples + i];
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
    snrt_fpu_fence();
    return;
}