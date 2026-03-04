// 2026 Luca Colombo Chips-IT
// Taken from the kernels folder, written by the people there reported!

#include "snrt.h"

#include "fastexpf.h"

void softmax_FP32(float *input, float *output, uint32_t row_stride, uint32_t batch_offset, uint32_t batch_size, uint32_t seq_len, uint32_t input_samples) {

    snrt_mcycle();

    float max_core, sum;

    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {

            max_core = (float)input[b * batch_offset + s * row_stride];  // initialize the max value of the current core
            sum = 0.0f;

            for (uint32_t i = 1; i < input_samples; i++) {
                if (input[b * batch_offset + s * row_stride + i] > max_core) {
                    max_core = (float) input[b * batch_offset + s * row_stride + i];
                }
            }

            // compute the shifted value of the current row
            for (uint32_t i = 0; i < input_samples; i++) {

                output[b * batch_offset + s * row_stride + i] = (float) fast_expf((float) (input[b * batch_offset + s * row_stride + i] - (float) max_core));
                sum += (float)output[b * batch_offset + s * row_stride + i];
            }

            // compute the softmax value of the current row
            for (uint32_t i = 0; i < input_samples; i++) {
                output[b * batch_offset + s * row_stride + i] /= (float) sum;
            }
        }
    }

    snrt_mcycle();

    return;
}