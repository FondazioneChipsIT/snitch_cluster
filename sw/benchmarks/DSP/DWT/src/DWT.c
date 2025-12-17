// Luca Colombo Chips-IT 2025
/* DWT filter */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "DWT_opt.h"

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        x = (double *)snrt_l1_next();
        y_low = x + LEN;
        y_high = y_low + LEN;
        h = y_high + LEN;
        g = h + FILTER_LEN;

        // If pointers are null -> break
        if (!x || !y_low || !y_high || !h || !g) {
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

        dwt_opt(chunk_per_core, offset, x, y_low, y_high, h, g);

    }

    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */