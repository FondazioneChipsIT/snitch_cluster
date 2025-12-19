// Luca Colombo Chips-IT 2025
/* CONV3 filter */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "conv3_opt.h"

bool use_opt = 1;

void conv3naive(uint32_t chunk_per_core, uint32_t offset,
    double *x, double *y, double *h){
    
    snrt_mcycle();

    for(uint32_t n = offset; n<offset+chunk_per_core; n++){
        double acc = 0.0;
        for(uint32_t i = 0 ; i<CONV3_LEN; i++){
            acc += x[n+i]*h[i];
        }
        y[n] = acc;
    }

    snrt_mcycle();

    return;
}

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
            
        }

        for(uint32_t i=0; i<CONV3_LEN;i++){
            h[i] = (double)(i);
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
            conv3_opt_V2(chunk_per_core, offset, x, y, h);
        else
            conv3naive(chunk_per_core, offset, x, y, h);

    }

    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */