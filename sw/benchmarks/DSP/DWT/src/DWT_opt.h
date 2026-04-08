#ifndef DWT_OPT_H
#define DWT_OPT_H

#include "snrt.h"
#include "data.h"

// ─────────────────────────────────────────────────────────────────────────────
//  dwt_opt — SSR + FREP optimised DWT kernel
//
//  Changes vs. original:
//   • in_len parameter added for right-boundary zero-padding
//   • start_k computed to skip out-of-bounds x accesses (right boundary)
//   • SSR base pointer and filter start adjusted by start_k
//   • Guard for effective_taps == 0 (full zero-pad output sample)
// ─────────────────────────────────────────────────────────────────────────────
static inline void dwt_opt(uint32_t chunk_per_core, uint32_t offset,
                            float *x, float *y_low, float *y_high,
                            float *h, float *g, uint32_t in_len){
    float zero = 0.0f;


    // Initialise accumulators ft3 (low) and ft4 (high) to 0.0
    asm volatile(
        "flw ft3, 0(%[z])\n"
        "flw ft4, 0(%[z])\n"
        : : [z] "r"(&zero) : "ft3", "ft4");

    for (uint32_t n = 0; n < chunk_per_core; n++) {

        uint32_t out    = offset + n;
        uint32_t center = 2 * out;

        // ── Boundary handling ───────────────────────────────────────────────
        //
        //  Left boundary (causal): when center < FILTER_LEN, only (center+1)
        //  past samples exist; additional taps would access x at negative index.
        //
        //  Right boundary (zero-pad): when center >= in_len, taps k=0,1,…
        //  would read x[center], x[center-1], … which are past the end of the
        //  array and treated as zero.  The first valid tap is:
        //      start_k  = max(0, center - in_len + 1)
        //
        //  After applying both constraints the effective convolution window is:
        //      k  ∈  [start_k,  start_k + eff_taps - 1]
        //      x accessed as x[center - k]  ∈  [0, in_len-1]  ✓

        uint32_t taps_left = (center + 1 < FILTER_LEN) ? (center + 1)
                                                        : FILTER_LEN;
        uint32_t start_k   = (center >= in_len) ? (center - in_len + 1) : 0;
        uint32_t eff_taps  = taps_left - start_k;

        // Entirely zero-padded sample — write 0 and move on
        if (__builtin_expect(eff_taps == 0, 0)) {
            y_low[out]  = 0.0f;
            y_high[out] = 0.0f;
            continue;
        }

        // ── SSR setup ───────────────────────────────────────────────────────
        //  DM0: x[center - start_k], x[center - start_k - 1], …  (stride -4)
        //  DM1: h[start_k],          h[start_k + 1],          …  (stride +4)

        snrt_ssr_loop_1d(SNRT_SSR_DM0, eff_taps, -(int)sizeof(float));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, eff_taps,  (int)sizeof(float));

        // ── Low-pass convolution ─────────────────────────────────────────────
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, x + center - start_k);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, h + start_k);

        snrt_ssr_enable();

        asm volatile(
            "frep.o %[r], 1, 0, 0\n"
            "fmadd.s ft3, ft0, ft1, ft3\n"
            : : [r] "r"(eff_taps - 1)
            : "ft0", "ft1", "ft3", "memory");

        // ── High-pass convolution (rewind DM0, switch DM1 to g) ─────────────
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, x + center - start_k);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, g + start_k);

        asm volatile(
            "frep.o %[r], 1, 0, 0\n"
            "fmadd.s ft4, ft0, ft1, ft4\n"
            : : [r] "r"(eff_taps - 1)
            : "ft0", "ft1", "ft4", "memory");

        // ── Store results and reset accumulators ─────────────────────────────
        asm volatile(
            "fsw    ft3, 0(%[yl])\n"
            "fsw    ft4, 0(%[yh])\n"
            "fsub.s ft3, ft3, ft3\n"   // ft3 = 0
            "fsub.s ft4, ft4, ft4\n"   // ft4 = 0
            : : [yl] "r"(y_low  + out),
                [yh] "r"(y_high + out)
            : "memory");

        snrt_ssr_disable();
        snrt_fpu_fence();
    }

}

#endif // DWT_OPT_H