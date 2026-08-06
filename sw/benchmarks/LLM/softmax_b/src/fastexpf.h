// 2026 Luca Colombo Chips-IT
#pragma once

#include <stdint.h>
// static const, not plain globals: as mutable globals they live in DRAM and
// the compiler has to reload them on every call (a store to the output array
// may alias them), which costs a DRAM access per element.
// The values are the same ones the original code used: the int operands were
// already converted to float by the C promotion rules before the multiply.
static const float fast_exp_coef = 12102203.0f;
static const float fast_exp_bias = 1064866805.0f;

static inline float fast_expf(float x) {
    union {
        float f;
        uint32_t i;
    } result;

    // Constants derived by Schraudolph
    result.i = (uint32_t)(fast_exp_coef * x + fast_exp_bias);
    // result.i = (uint32_t)(12102203 * x + 1064866805);
    return result.f;
}
