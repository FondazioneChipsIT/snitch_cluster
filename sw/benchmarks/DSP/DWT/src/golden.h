#ifndef GOLDEN_H
#define GOLDEN_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "data.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Tolerance for float comparison
// ─────────────────────────────────────────────────────────────────────────────
#define GOLDEN_ABS_TOL 1e-4f
#define GOLDEN_REL_TOL 1e-5f

// ─────────────────────────────────────────────────────────────────────────────
//  Golden buffers (static, allocated in BSS — kept off TCDM)
// ─────────────────────────────────────────────────────────────────────────────
static float gold_low [DWT_LEVELS][OUT_LEN_0];   // OUT_LEN_0 is the largest
static float gold_high[DWT_LEVELS][OUT_LEN_0];

// ─────────────────────────────────────────────────────────────────────────────
//  Single-level reference DWT
//
//  Fully equivalent to what dwt_opt / dwt_naive should produce:
//    • causal left  boundary  → truncate taps
//    • zero-pad right boundary → skip invalid x accesses
// ─────────────────────────────────────────────────────────────────────────────
static void golden_dwt_level(const float *in,  uint32_t in_len,
                              float *out_low,   float *out_high,
                              uint32_t out_len,
                              const float *fh,  const float *fg)
{
    for (uint32_t n = 0; n < out_len; n++) {

        uint32_t center = 2 * n;

        // Left boundary
        uint32_t taps_left = (center + 1 < FILTER_LEN) ? (center + 1)
                                                        : FILTER_LEN;
        // Right boundary
        uint32_t start_k   = (center >= in_len) ? (center - in_len + 1) : 0;
        uint32_t eff_taps  = taps_left - start_k;

        float acc_low  = 0.0f;
        float acc_high = 0.0f;

        for (uint32_t k = start_k; k < start_k + eff_taps; k++) {
            float s   = in[center - k];
            acc_low  += s * fh[k];
            acc_high += s * fg[k];
        }

        out_low [n] = acc_low;
        out_high[n] = acc_high;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Compute all 4 levels of the golden reference.
//  Call this AFTER the DM core has initialised x, h, g.
// ─────────────────────────────────────────────────────────────────────────────
static void golden_compute(void)
{
    for (int level = 0; level < DWT_LEVELS; level++) {

        const float *in  = (level == 0) ? x : gold_low[level - 1];
        uint32_t  in_len = dwt_in_len [level];
        uint32_t out_len = dwt_out_len[level];

        golden_dwt_level(in, in_len,
                         gold_low [level],
                         gold_high[level],
                         out_len,
                         h, g);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Comparison helpers
// ─────────────────────────────────────────────────────────────────────────────
static inline bool nearly_equal(float a, float b)
{
    float diff = fabsf(a - b);
    if (diff <= GOLDEN_ABS_TOL) return true;
    float mag = fmaxf(fabsf(a), fabsf(b));
    return diff <= GOLDEN_REL_TOL * mag;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Verify one output vector against its golden counterpart.
//  Returns the number of mismatches (0 = pass).
// ─────────────────────────────────────────────────────────────────────────────
static uint32_t golden_check_vector(const float *got,
                                    const float *ref,
                                    uint32_t     len,
                                    const char  *name,
                                    int          level)
{
    uint32_t errors = 0;
    float    max_abs_err = 0.0f;
    float    max_rel_err = 0.0f;
    uint32_t first_err_idx = 0;

    for (uint32_t i = 0; i < len; i++) {
        if (!nearly_equal(got[i], ref[i])) {
            if (errors == 0) first_err_idx = i;
            errors++;
        }
        float ae = fabsf(got[i] - ref[i]);
        float re = (fabsf(ref[i]) > 1e-10f) ? ae / fabsf(ref[i]) : ae;
        if (ae > max_abs_err) max_abs_err = ae;
        if (re > max_rel_err) max_rel_err = re;
    }

    return errors;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Top-level verification: compare all levels of low[] and high[].
//  Returns total mismatch count across all levels and both banks.
//  Call this from the DM core after all compute cores have finished.
// ─────────────────────────────────────────────────────────────────────────────
static uint32_t golden_verify(void)
{
    uint32_t total_errors = 0;

    for (int level = 0; level < DWT_LEVELS; level++) {
        uint32_t len = dwt_out_len[level];
        total_errors += golden_check_vector(low [level], gold_low [level],
                                            len, "low",  level);
        total_errors += golden_check_vector(high[level], gold_high[level],
                                            len, "high", level);
    }

    return total_errors;
}

#endif // GOLDEN_H