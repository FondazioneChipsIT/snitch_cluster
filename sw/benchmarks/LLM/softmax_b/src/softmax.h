// 2026 Luca Colombo Chips-IT
// Taken from the kernels folder, written by the people there reported!

#include "snrt.h"

#include "fastexpf.h"
// These are written by one core and read by another with only a barrier in
// between. As plain globals they live in DRAM, and a store to DRAM is posted:
// it can still be in flight when the barrier releases, so the reader may see
// the previous value. Whether it works depends on the timing, which is exactly
// the kind of bug that moves when anything else changes. They are allocated in
// TCDM next to the barrier instead (same memory as the barrier => a store
// issued before the barrier has landed when the barrier completes).
float *max_core, *sum;      // [8] each
float *core0max, *global_sum;
// The barrier must NOT be a plain global: globals are linked into L3 (the
// linker script only has the DRAM region), so every snrt_partial_barrier would
// spin on a DRAM address at ~60 cycles per access. It is allocated in TCDM by
// the DM core in main instead, and this global only holds the pointer.
snrt_barrier_t *barr;

// NOTE on the parameters: input/output already point at the slice of this
// core and input_samples is the size of that slice, but two consecutive rows
// are still row_stride elements apart. Before, input_samples was used as the
// row stride too, which gave wrong addresses from the second row on (harmless
// only because this data set has BATCH_SIZE = SEQ_LEN = 1).
//
// NOTE on speed: max_core[] and sum[] are globals, so they live in DRAM.
// Accumulating directly into them inside the loops meant one DRAM
// read-modify-write per element (~60 cycles each). They are now written once
// per row and the accumulation is done in a local variable, which the
// compiler keeps in a register.
void softmax_FP32(float *input, float *output, uint32_t batch_size,
                  uint32_t seq_len, uint32_t input_samples,
                  uint32_t row_stride) {

    uint32_t core_idx = snrt_cluster_core_idx();

    // Local copy: otherwise the global pointer is re-read from DRAM after
    // every barrier call (the call writes memory, so it cannot be cached)
    snrt_barrier_t *bar_p = barr;

    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t s = 0; s < seq_len; s++) {

            float *in_row  = input  + b * seq_len * row_stride + s * row_stride;
            float *out_row = output + b * seq_len * row_stride + s * row_stride;

            // max of this core's slice, kept in a register
            float local_max = in_row[0];
            for (uint32_t i = 1; i < input_samples; i++) {
                if (in_row[i] > local_max) {
                    local_max = in_row[i];
                }
            }
            max_core[core_idx] = local_max;

            snrt_partial_barrier(bar_p, 8);

            if (core_idx == 0) {
                *core0max = max_core[0];
                for (uint32_t i = 1; i < 8; i++) {
                    if (max_core[i] > *core0max) {
                        *core0max = max_core[i];
                    }
                }
            }

            snrt_partial_barrier(bar_p, 8);

            // exp and running sum, again with a local accumulator
            float mx = *core0max;
            float local_sum = 0.0f;
            for (uint32_t i = 0; i < input_samples; i++) {
                float e = fast_expf(in_row[i] - mx);
                out_row[i] = e;
                local_sum += e;
            }
            sum[core_idx] = local_sum;

            snrt_partial_barrier(bar_p, 8);

            if (core_idx == 0) {
                float gsum = 0.0f;
                for (uint32_t c = 0; c < 8; c++)
                    gsum += sum[c];
                *global_sum = gsum;
            }

            snrt_partial_barrier(bar_p, 8);

            // one reciprocal and then multiplies, instead of one fdiv per
            // element (fdiv is not pipelined, fmul is)
            float inv = 1.0f / *global_sum;
            for (uint32_t i = 0; i < input_samples; i++) {
                out_row[i] *= inv;
            }

            snrt_partial_barrier(bar_p, 8);
        }
    }
    snrt_fpu_fence();
    return;
}
