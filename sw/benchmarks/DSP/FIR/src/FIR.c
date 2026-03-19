// Luca Colombo Chips-IT 2025
/* FIR filter */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "fir_opt.h"

void fir_naive(uint32_t chunk_per_core, uint32_t offset,
               float *x, float *y, float *h){
    snrt_mcycle();
    for (uint32_t n = offset; n < offset + chunk_per_core; n++) {

        float acc = 0.0f;

        /* Gestione del bordo sinistro */
        uint32_t taps = (n < FILTER_LEN) ? (n + 1) : FILTER_LEN;

        for (uint32_t k = 0; k < taps; k++) {
            acc += x[n - k] * h[k];
        }

        y[n] = acc;
    }
    snrt_mcycle();
}

bool use_opt = 1;

uint32_t CHECK_RESULTS = 1;

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        x = (float *)snrt_l1_next();
        y = x + LEN;
        h = y + LEN;

        size_t size = LEN * sizeof(float);
        size_t size_filter = FILTER_LEN * sizeof(float);

        snrt_dma_start_1d(x, x_data, size);
        snrt_dma_start_1d(h, h_data, size_filter);
        snrt_dma_wait_all();

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the vector per core, and the offset that is used to space them
        uint32_t chunk_per_core = (LEN-FILTER_LEN)/ncores; 
 
        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx * chunk_per_core + (FILTER_LEN - 1);
        
        // Call the kernel
        if(use_opt)
            fir_opt(chunk_per_core, offset, x, y, h);
        else    
            fir_naive(chunk_per_core, offset, x, y, h);

    }

    return 0;
}