// Luca Colombo Chips-IT 2025
/* CONV3x3 2d filter with no padding */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "conv3x3_opt.h"

bool use_opt = 1;

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();
    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        x = (double *)snrt_l1_next();
        y = x + LEN*LEN;
        // Y without padding has less elements
        h = y + (LEN-2)*(LEN-2);

        // If pointers are null -> break
        if (!x || !y || !h) {
            printf("Memory allocation failed!\n");
            return -1;
        } 

        // Initialize the values of vectors, can change as you like
        for(uint32_t i = 0; i<LEN*LEN; i++){
            x[i] = (double)i;
            
        }

        for(uint32_t i=0; i<CONV3x3_LEN*CONV3x3_LEN;i++){
            h[i] = (double)(i);
        }

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the matrix, represents
        // the number of rows each core has to use
        uint32_t chunk_per_core = LEN/ncores;

        if(chunk_per_core == 0){
            printf("Chunk for each core is 0!\n");
            return -2;
        }
        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx*chunk_per_core;
        
        // Call the kernel
        conv3x3_opt(core_idx,chunk_per_core, offset, x, y, h);

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    /*if(core_idx==0){

        // Do not read last two cols and rows, are not used
        for(uint32_t i = 0; i<LEN-2; i++){
            printf("Row %d: ",i);
            for(uint32_t j = 0; j<LEN-2; j++){
                printf(" %.1f ", y[i*(LEN-2)+j]);
            }
            printf("\n");
        }


    }*/

    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */