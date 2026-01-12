// Luca Colombo Chips-IT 2025
/* Residual that adds original to result of CONV3x3 */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "residual_opt.h"

void residual_naive(uint32_t core_idx, uint32_t chunk_per_core, uint32_t offset,
    double *x, double *y, double *out){
    
    snrt_mcycle();

    // Chunk per core is the number of rows, we need the total amount
    uint32_t tot_ops = chunk_per_core*LEN;

    for (uint32_t i = offset; i<offset + tot_ops; i++){

        out[i] = x[i]+y[i];

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
        out = x + LEN*LEN;
        y = out + LEN*LEN;

        // If pointers are null -> break
        if (!x || !y || !out) {
            printf("Memory allocation failed!\n");
            return -1;
        } 

        // Initialize the values of vectors, can change as you like
        for(uint32_t i = 0; i<LEN*LEN; i++){
            x[i] = (double)i;
            y[i] = (double)i;
        }

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the matrix, represents
        // the number of rows each core has to use
        uint32_t chunk_per_core = LEN/ncores;
        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx*chunk_per_core;

        // We call the residual kernel
        if(use_opt)
            residual_opt(core_idx,chunk_per_core, offset, x, y, out);
        else
            residual_naive(core_idx,chunk_per_core, offset, x, y, out);
    }

    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */