// Luca Colombo Chips-IT 2025
/* DWT filter */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "DWT_opt.h"

void dwt_naive(uint32_t chunk_per_core, uint32_t offset,
               float *x, float *y_low, float *y_high,
               float *h, float *g){
    snrt_mcycle();
    for (uint32_t n = 0; n < chunk_per_core; n++) {

        uint32_t out = offset + n;
        uint32_t center = 2 * out;

        /* Numero di tap effettivi (gestione bordo sinistro) */
        uint32_t taps = (center < FILTER_LEN) ? (center + 1) : FILTER_LEN;

        float acc_low  = 0.0f;
        float acc_high = 0.0f;

        /*
         * Convoluzione:
         * x[center - k] * h[k]
         * x[center - k] * g[k]
         */
        for (uint32_t k = 0; k < taps; k++) {
            float sample = x[center - k];
            acc_low  += sample * h[k];
            acc_high += sample * g[k];
        }

        y_low[out]  = acc_low;
        y_high[out] = acc_high;
    }
    snrt_mcycle();
}

bool use_opt = 1;

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        x = (float *)snrt_l1_next();
        y_low = x + LEN;
        y_high = y_low + LEN/2;
        h = y_high + LEN/2;
        g = h + FILTER_LEN;

        // If pointers are null -> break
        if (!x || !y_low || !y_high || !h || !g) {
            printf("Memory allocation failed!\n");
            return -1;
        } 

        // Initialize the values of vectors, can change as you like
        for(uint32_t i = 0; i<LEN; i++){
            x[i] = (float)i;
        }
        /* Haar
        h[0] = 1.0;
        h[1] = 0.0;

        g[0] = 0.0;
        g[1] = 1.0;*/

        /* DB4

        h[0] =  0.4829629131445341;
        h[1] =  0.8365163037378079;
        h[2] =  0.2241438680420134;
        h[3] = -0.1294095225512604;

        g[0] = -0.1294095225512604;
        g[1] = -0.2241438680420134;
        g[2] =  0.8365163037378079;
        g[3] = -0.4829629131445341;*/

        // Daubechies-16 low-pass (analysis)
        size_t size = FILTER_LEN * sizeof(float);
        snrt_dma_start_1d(h, h_L2, size);
        snrt_dma_wait_all();
        
        // High pass
        for (int k = 0; k < 32; k++) {
            g[k] = ((k & 1) ? -1.0f : 1.0f) * h[31 - k];
        }


    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the vector per core, and the offset that is used to space them
        uint32_t dwt_len = LEN / 2;
        uint32_t chunk_per_core = dwt_len / ncores;
        uint32_t offset = core_idx * chunk_per_core;

        // Call the kernel
        if(use_opt)
            dwt_opt(chunk_per_core, offset, x, y_low, y_high, h, g);
        else
            dwt_naive(chunk_per_core, offset, x, y_low, y_high, h, g);
    }



    return 0;
}

