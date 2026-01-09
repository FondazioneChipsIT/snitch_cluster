// Luca Colombo Chips-IT 2025
/* Residual that adds original to result of CONV3x3 */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "conv3x3_opt.h"
#include "residual_opt.h"

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
        h = y + LEN*LEN;

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
        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx*chunk_per_core;
        
        // We first compute the convolution
        conv3x3_opt(core_idx,chunk_per_core, offset, x, y, h);

        //Padding
        if(core_idx==0){
            for(uint32_t i=0; i<LEN-2;i++){
                for(uint32_t j= LEN-2; j<LEN;j++){
                    y[i*LEN + j]= 0.1;
                }

            }
            for(uint32_t i=LEN-2; i<LEN;i++){
                for(uint32_t j= 0; j<LEN;j++){
                    y[i*LEN + j ]= 0.1;
                }
            }
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
        residual_opt(core_idx,chunk_per_core, offset, x, y, out);

    }

    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */