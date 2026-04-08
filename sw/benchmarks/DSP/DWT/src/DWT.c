// Luca Colombo — Chips-IT 2025
// DWT benchmark: 40-tap filter, 1000 inputs, 1155 outputs, 4 decomposition levels

#include "snrt.h"
#include "data.h"
#include "DWT_opt.h"
#include "golden.h"

bool use_opt = 1;

int main(){
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores   = snrt_cluster_compute_core_num();

    // ── DM core: allocate TCDM and initialise data ──────────────────────────
    if (snrt_is_dm_core()) {

        float *mem = (float *)snrt_l1_next();

        // Input signal
        x = mem; 
        mem += LEN;

        // Low/high output buffers for each decomposition level
        for (int j = 0; j < DWT_LEVELS; j++) {
            low[j]  = mem; 
            mem += dwt_out_len[j];
            high[j] = mem; 
            mem += dwt_out_len[j];
        }

        // Analysis filters
        h = mem; 
        mem += FILTER_LEN;
        g = mem; 
        mem += FILTER_LEN;

        // Initialise input: x[i] = i
        for (uint32_t i = 0; i < LEN; i++)
            x[i] = (float)i;

        // Low-pass: box filter  h[k] = 1/FILTER_LEN
        for (int i = 0; i < FILTER_LEN; i++)
            h[i] = 1.0f / FILTER_LEN;

        // High-pass: quadrature mirror  g[k] = (-1)^k * h[L-1-k]
        for (int i = 0; i < FILTER_LEN; i++)
            g[i] = ((i & 1) ? -1.0f : 1.0f) * h[FILTER_LEN - 1 - i];

        // Check golden
        golden_compute();
    }

    snrt_cluster_hw_barrier(); 

    // ── Compute cores: 4-level DWT ──────────────────────────────────────────
    // Each level: parallelise the out_len[j] output samples across ncores.
    // A cluster barrier between levels ensures low[j] is fully written before
    // it is used as input to level j+1.

    snrt_mcycle();
    for (int level = 0; level < DWT_LEVELS; level++) {

        if (snrt_is_compute_core()) {

            uint32_t cur_in  = dwt_in_len[level];
            uint32_t cur_out = dwt_out_len[level];

            // Input for this level: original x at level 0, low-pass of
            // previous level afterwards
            float *cur_x = (level == 0) ? x : low[level - 1];

            // Work-sharing: distribute cur_out samples across cores.
            // Remainder samples are assigned to the first `rem` cores
            // (each gets one extra), so every sample is covered.
            uint32_t chunk  = cur_out / ncores;
            uint32_t rem    = cur_out % ncores;
            uint32_t core_chunk  = chunk + (core_idx < rem ? 1u : 0u);
            uint32_t offset = core_idx * chunk
                                 + (core_idx < rem ? core_idx : rem);
           
            dwt_opt  (core_chunk, offset, cur_x,
                        low[level], high[level], h, g, cur_in);
          
        }

        // Barrier between levels — DM core participates to keep cluster in sync
        snrt_cluster_hw_barrier();
    }
    snrt_mcycle();

    if(core_idx == 0) {
        // Check results
        uint32_t errs = golden_verify();
        return (errs == 0) ? 0 : -1;
    }

    return 0;
}