// Luca Colombo Chips-IT 2025
/* FIR filter */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "fir_opt.h"

void fir_naive(uint32_t chunk_per_core, uint32_t offset,
               double *x, double *y, double *h){
    snrt_mcycle();
    for (uint32_t n = offset; n < offset + chunk_per_core; n++) {

        double acc = 0.0;

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

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        x = (double *)snrt_l1_next();
        y = x + LEN;
        h = y + LEN;

        // If pointers are null -> break
        if (!x || !y || !h) {
            printf("Memory allocation failed!\n");
            return -1;
        } 

        // Initialize the values of vectors, can change as you like
        for(uint32_t i = 0; i<LEN; i++){
            x[i] = (double)i;
            h[i] = (double)(LEN-i);
        }

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the vector per core, and the offset that is used to space them
        uint32_t chunk_per_core = LEN/ncores;
        if(chunk_per_core == 0){
            printf("Chunk for each core is 0!\n");
            return -2;
        }
        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx*chunk_per_core;
        
    
        // Call the kernel
        if(use_opt)
            fir_opt(chunk_per_core, offset, x, y, h);
        else    
            fir_naive(chunk_per_core, offset, x, y, h);

    }

    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */