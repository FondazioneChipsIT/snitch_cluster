#ifndef DATA_H
#define DATA_H

// ──── Benchmark parameters ──────────────────────────────────────────────────
#ifndef LEN
#define LEN        1000   // input samples
#endif

#ifndef FILTER_LEN
#define FILTER_LEN  40    // 40-coefficient filter
#endif

#define DWT_LEVELS   4

// ──── Output lengths per level ──────────────────────────────────────────────
// out_len[j] = floor((in_len[j] + FILTER_LEN - 1) / 2)
// Level 0: (1000 + 39) / 2 = 519
// Level 1: ( 519 + 39) / 2 = 279
// Level 2: ( 279 + 39) / 2 = 159
// Level 3: ( 159 + 39) / 2 = 99
// Total outputs (DWT coefficients): 519 + 279 + 159 + 99 + 99 = 1155
#define OUT_LEN_0  ((LEN       + FILTER_LEN - 1) / 2)   // 519
#define OUT_LEN_1  ((OUT_LEN_0 + FILTER_LEN - 1) / 2)   // 279
#define OUT_LEN_2  ((OUT_LEN_1 + FILTER_LEN - 1) / 2)   // 159
#define OUT_LEN_3  ((OUT_LEN_2 + FILTER_LEN - 1) / 2)   // 99

#define TOTAL_OUTPUTS \
    (OUT_LEN_0 + OUT_LEN_1 + OUT_LEN_2 + OUT_LEN_3 + OUT_LEN_3)  // 1155

// ──── Global TCDM pointers ──────────────────────────────────────────────────
float *x;                      // original input  [LEN]
float *low[DWT_LEVELS];        // low-pass outputs [OUT_LEN_j]
float *high[DWT_LEVELS];       // high-pass outputs[OUT_LEN_j]
float *h, *g;                  // analysis filters [FILTER_LEN]

// ──── Compile-time lookup tables (used in main) ─────────────────────────────
static const uint32_t dwt_in_len[DWT_LEVELS]  = {LEN,      OUT_LEN_0,
                                                  OUT_LEN_1, OUT_LEN_2};
static const uint32_t dwt_out_len[DWT_LEVELS] = {OUT_LEN_0, OUT_LEN_1,
                                                  OUT_LEN_2, OUT_LEN_3};

#endif // DATA_H