#pragma once

// Floating-point multiplications by zero cannot be optimized as in some
// edge cases they do not yield zero:
// - 0f * NaN = NaN
// - 0f * INFINITY == NaN
// Thus in order to optimize it, we need to test for zero. You can use this
// function for free when `multiplier` is a constant.

static inline float multiply_opt(float multiplicand, float multiplier) {
    if (multiplier)
        return multiplicand * multiplier;
    else
        return 0.0f;
}


#include "gemm/src/gemm.h"
#include "gemv/src/gemv.h"
