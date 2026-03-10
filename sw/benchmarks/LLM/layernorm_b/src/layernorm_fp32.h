// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Modified: scalar-only version (no vf*.s SIMD instructions)

#include "snrt.h"

#define UNROLL 4

static inline void layernorm_fp32_opt(float *input, float *output,
                                      uint32_t batch_size, uint32_t seq_len,
                                      const uint32_t embeddings, int32_t eps) {
    if (snrt_is_compute_core()) {
        uint32_t offset = snrt_cluster_core_idx() * embeddings;
        uint32_t stride = snrt_cluster_compute_core_num() * embeddings;
        uint32_t tile_seq_len = seq_len / snrt_cluster_compute_core_num();
        float *core_itile = input + offset;
        float *core_otile = output + offset;

        uint32_t batch_offset = seq_len * embeddings;

        // compute the mean and variance along the last dimension
        float mean_tot = 0.0f;
        float var_tot = 0.0f;
        float mean_reg = 0.0f;

        for (int32_t b = 0; b < batch_size; b++) {

            // ------------------------------------------------------------------
            // SSR loop bounds / strides (4-D for DM0/DM1, 2-D for DM2)
            // Inner stride: sizeof(float)
            // Second stride: UNROLL * sizeof(float)
            // ------------------------------------------------------------------
            const uint32_t ssr0_b[4] = {
                UNROLL,
                embeddings / UNROLL,
                2,
                tile_seq_len};
            const uint32_t ssr0_i[4] = {
                sizeof(float),          
                UNROLL * sizeof(float),
                0,
                stride * sizeof(float)};

            const uint32_t ssr1_b[2] = {
                UNROLL,
                embeddings / UNROLL};
            const uint32_t ssr1_i[2] = {
                sizeof(float),          
                UNROLL * sizeof(float)};

            snrt_ssr_loop_4d(SNRT_SSR_DM0, ssr0_b[0], ssr0_b[1], ssr0_b[2],
                             ssr0_b[3], ssr0_i[0], ssr0_i[1], ssr0_i[2],
                             ssr0_i[3]);

            snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr0_b[0], ssr0_b[1], ssr0_b[2],
                             ssr0_b[3], ssr0_i[0], ssr0_i[1], ssr0_i[2],
                             ssr0_i[3]);

            snrt_ssr_loop_2d(SNRT_SSR_DM2, ssr1_b[0], ssr1_b[1], ssr1_i[0],
                             ssr1_i[1]);

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D,
                          &core_itile[b * batch_offset]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_4D,
                           &core_otile[b * batch_offset]);

            // Each frep iteration now handles UNROLL scalars (was UNROLL*2 with SIMD)
            const uint32_t n_frep = embeddings / UNROLL; // was embeddings/(UNROLL*2)

            for (int32_t s = 0; s < tile_seq_len; s++) {
                float mean[UNROLL] = {0.0f, 0.0f, 0.0f, 0.0f};
                float var[UNROLL]  = {0.0f, 0.0f, 0.0f, 0.0f};
                mean_tot = 0.0f;
                var_tot  = 0.0f;

                // Scalar registers replacing v2f32 vectors
                float var_reg[UNROLL];
                float pow_reg[UNROLL];
                float one_reg = 1.0f; // was v2f32 one_reg = {1.0f, 1.0f}

                snrt_ssr_enable();

                // --------------------------------------------------------------
                // Pass 1 – mean of the row
                // Original: vfcpka.s.s + frep(vfsum.s)
                // Scalar:   fmv.s      + frep(fadd.s)
                //
                // Each frep iteration reads UNROLL scalars from ft0 (SSR DM0)
                // and accumulates into mean[0..3].
                // After the loop the four partial sums are reduced and divided.
                // --------------------------------------------------------------
                asm volatile(
                    // Zero the four accumulators (was vfcpka.s.s ... %[zero],%[zero])
                    "fmv.s %[mean0], %[zero] \n"
                    "fmv.s %[mean1], %[zero] \n"
                    "fmv.s %[mean2], %[zero] \n"
                    "fmv.s %[mean3], %[zero] \n"
                    // FREP: 4 fadd.s replace 4 vfsum.s
                    "frep.o  %[n_frep], 4, 0, 0 \n"
                    "fadd.s %[mean0], %[mean0], ft0 \n"
                    "fadd.s %[mean1], %[mean1], ft0 \n"
                    "fadd.s %[mean2], %[mean2], ft0 \n"
                    "fadd.s %[mean3], %[mean3], ft0 \n"
                    // Horizontal reduction (unchanged, already scalar)
                    "fadd.s %[mean0], %[mean0], %[mean1] \n"
                    "fadd.s %[mean2], %[mean2], %[mean3] \n"
                    "fadd.s %[mean_tot], %[mean0], %[mean2] \n"
                    "fdiv.s %[mean_tot], %[mean_tot], %[embeddings] \n"
                    : [mean0]    "+f"(mean[0]),
                      [mean1]    "+f"(mean[1]),
                      [mean2]    "+f"(mean[2]),
                      [mean3]    "+f"(mean[3]),
                      [mean_tot] "+f"(mean_tot)
                    : [n_frep]     "r"(n_frep - 1),
                      [zero]       "f"(0.0f),
                      [embeddings] "f"((float)embeddings)
                    : "ft0", "ft1", "ft2");

                snrt_fpu_fence();

                // --------------------------------------------------------------
                // Pass 2 – variance (and write centred values to DM1)
                //
                // Original: vfcpka.s.s / vfsub.s / vfadd.s / vfmul.s / vfsum.s
                // Scalar:   fmv.s      / fsub.s  / fadd.s  / fmul.s  / fadd.s
                //
                // var[0..3] must be pre-zeroed before frep because fadd.s is
                // used as an accumulator (vfsum.s already was).
                // --------------------------------------------------------------
                asm volatile(
                    // Broadcast mean_tot into mean_reg (was vfcpka.s.s %[mean_reg], %[mean_tot], %[mean_tot])
                    "fmv.s %[mean_reg], %[mean_tot] \n"
                    // Zero the variance accumulators
                    "fmv.s %[var0], %[zero] \n"
                    "fmv.s %[var1], %[zero] \n"
                    "fmv.s %[var2], %[zero] \n"
                    "fmv.s %[var3], %[zero] \n"
                    // FREP body – 16 instructions, same count as original
                    "frep.o  %[n_frep], 16, 0, 0 \n"
                    // (x - mean) → var_reg[0..3]   (was vfsub.s)
                    "fsub.s %[var_reg0], ft0, %[mean_reg] \n"
                    "fsub.s %[var_reg1], ft0, %[mean_reg] \n"
                    "fsub.s %[var_reg2], ft0, %[mean_reg] \n"
                    "fsub.s %[var_reg3], ft0, %[mean_reg] \n"
                    // Write centred values to DM1 output stream (was vfadd.s ft1, ..., %[zero])
                    "fadd.s ft1, %[var_reg0], %[zero] \n"
                    "fadd.s ft1, %[var_reg1], %[zero] \n"
                    "fadd.s ft1, %[var_reg2], %[zero] \n"
                    "fadd.s ft1, %[var_reg3], %[zero] \n"
                    // (x - mean)^2  (was vfmul.s)
                    "fmul.s %[pow0], %[var_reg0], %[var_reg0] \n"
                    "fmul.s %[pow1], %[var_reg1], %[var_reg1] \n"
                    "fmul.s %[pow2], %[var_reg2], %[var_reg2] \n"
                    "fmul.s %[pow3], %[var_reg3], %[var_reg3] \n"
                    // Accumulate into var[0..3]  (was vfsum.s which also summed packed pair)
                    "fadd.s %[var0], %[var0], %[pow0] \n"
                    "fadd.s %[var1], %[var1], %[pow1] \n"
                    "fadd.s %[var2], %[var2], %[pow2] \n"
                    "fadd.s %[var3], %[var3], %[pow3] \n"
                    // Post-frep: horizontal reduction + normalise (unchanged)
                    "fadd.s %[var0], %[var0], %[var1] \n"
                    "fadd.s %[var2], %[var2], %[var3] \n"
                    "fadd.s %[var_tot], %[var0], %[var2] \n"
                    "fdiv.s %[var_tot], %[var_tot], %[embeddings] \n"
                    "fadd.s %[var_tot], %[var_tot], %[eps] \n"
                    "fsqrt.s %[var_tot], %[var_tot] \n"
                    "fdiv.s %[var_tot], %[one_reg], %[var_tot] \n"
                    // Keep 1/std in mean_reg for the normalisation pass
                    // (was vfcpka.s.s %[mean_reg], %[var_tot], %[var_tot])
                    "fmv.s %[mean_reg], %[var_tot] \n"

                    : [var_reg0] "+f"(var_reg[0]),
                      [var_reg1] "+f"(var_reg[1]),
                      [var_reg2] "+f"(var_reg[2]),
                      [var_reg3] "+f"(var_reg[3]),
                      [pow0]     "+f"(pow_reg[0]),
                      [pow1]     "+f"(pow_reg[1]),
                      [pow2]     "+f"(pow_reg[2]),
                      [pow3]     "+f"(pow_reg[3]),
                      [var0]     "+f"(var[0]),
                      [var1]     "+f"(var[1]),
                      [var2]     "+f"(var[2]),
                      [var3]     "+f"(var[3]),
                      [var_tot]  "+f"(var_tot),
                      [mean_reg] "+f"(mean_reg)
                    : [n_frep]     "r"(n_frep - 1),
                      [mean_tot]   "f"(mean_tot),
                      [embeddings] "f"((float)embeddings),
                      [eps]        "f"((float)eps),
                      [zero]       "f"(0.0f),
                      [one_reg]    "f"(one_reg)
                    : "ft0", "ft1", "ft2");

                snrt_fpu_fence();

                snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D,
                              &core_otile[b * batch_offset + s * stride]);

                // --------------------------------------------------------------
                // Pass 3 – normalisation:  out = centred * (1/std)
                // ft2 reads centred values from DM2, ft1 writes to DM1.
                // Original: vfmul.s ft1, ft2, %[mean_reg]
                // Scalar:   fmul.s  ft1, ft2, %[mean_reg]   (identical structure)
                // --------------------------------------------------------------
                asm volatile(
                    "frep.o  %[n_frep], 4, 0, 0 \n"
                    "fmul.s ft1, ft2, %[mean_reg] \n"
                    "fmul.s ft1, ft2, %[mean_reg] \n"
                    "fmul.s ft1, ft2, %[mean_reg] \n"
                    "fmul.s ft1, ft2, %[mean_reg] \n"
                    : [mean_reg] "+f"(mean_reg)
                    : [n_frep] "r"(n_frep - 1)
                    : "ft0", "ft1", "ft2");

                snrt_ssr_disable();
            }
        }

        snrt_fpu_fence();
    }
}