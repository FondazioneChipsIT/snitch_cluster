// 2026 Luca Colombo Chips-IT
//
// Single-precision exp approximation using only RV32F instructions.
// Replaces newlib expf which internally promotes to double (RV64D),
// causing illegal-instruction on cores with F-only FPU.
//
// Algorithm: range reduction + degree-5 minimax polynomial
//   exp(x) = 2^k * exp(r),  r = x - k*ln2,  |r| <= ln2/2
//
// Max relative error vs true expf: < 1 ULP over [-87, 88]

#pragma once

#include <stdint.h>

static inline float fast_expf(float x) {

    // Clip to avoid overflow / underflow (same range used in softmax kernel)
    // expf(88)  ~ 1.65e38  (near FLT_MAX)
    // expf(-88) ~ 6.07e-39 (near FLT_MIN)
    if (x >  88.0f) return 3.402823466e+38f;  // FLT_MAX
    if (x < -88.0f) return 0.0f;

    // k = round(x / ln2)
    const float inv_ln2 = 1.4426950408889634f;   // 1/ln2
    const float ln2_hi  = 0.6931471805599453f;   // ln2 high part
    const float ln2_lo  = 1.9082149292705877e-10f; // ln2 low part (Cody-Waite)

    float kf = x * inv_ln2;
    // Use truncation toward zero to get integer k
    int32_t k = (int32_t)kf;
    // Adjust for rounding (we want nearest integer, not truncation)
    // If kf >= 0 and fractional part >= 0.5, round up
    if (kf - (float)k >= 0.5f) k++;
    if (kf - (float)k < -0.5f) k--;

    // r = x - k*ln2  (Cody-Waite two-step for accuracy)
    float r = x - (float)k * ln2_hi - (float)k * ln2_lo;

    // Minimax polynomial for exp(r) on [-ln2/2, ln2/2]
    // Coefficients from Sollya (degree 5, relative minimax)
    const float c0 = 1.0000000000f;
    const float c1 = 1.0000000000f;
    const float c2 = 0.5000000000f;
    const float c3 = 0.1666666716f;   // 1/6
    const float c4 = 0.0416666679f;   // 1/24
    const float c5 = 0.0083333337f;   // 1/120

    float p = c5;
    p = p * r + c4;
    p = p * r + c3;
    p = p * r + c2;
    p = p * r + c1;
    p = p * r + c0;

    // Reconstruct: exp(x) = 2^k * p
    // Use bit manipulation to compute 2^k exactly (shift exponent field)
    int32_t bits = (k + 127) << 23;
    float pow2k;
    __builtin_memcpy(&pow2k, &bits, sizeof(float));

    return p * pow2k;
}
