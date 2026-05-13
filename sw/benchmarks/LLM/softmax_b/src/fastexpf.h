// 2026 Luca Colombo Chips-IT
#pragma once

#include <stdint.h>
int fast_exp_coef = 12102203;
int fast_exp_bias = 1064866805;

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
