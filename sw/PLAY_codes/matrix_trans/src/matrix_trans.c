// Luca Colombo Chips-IT 2025
/* Does transpose of matrix src and copies in matrix dst */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "matrix_trans_opt.h"

#include <stdio.h>

// Print first 4 and last 4 results
bool PRINT_RESULTS = 1;

// Possibility to run single core, useful to compare
// to new SIMD shuffle instruction (64x64, single core)
bool single_core = 1;

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores;

    if(single_core)
        ncores = 1;
    else
        ncores = snrt_cluster_compute_core_num();

    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        src = (double *)snrt_l1_next();
        dst = src + elems*elems; // The matrix is put in L1 by rows, so we need to allocate
        // rows*cols elements

        // If pointers are null -> break
        if (!src || !dst) {
            printf("Memory allocation failed!\n");
            return -1;
        }
        
        for(int i = 0; i < elems*elems; i++){
            src[i] = (double)i;
        }
    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        if(single_core && core_idx>0){
            // Do nothing cores from 1 to 7 if single core
        }
        else{
            // Compute the chunk of the vector per core, and the offset that is used to space them
            uint32_t chunk_per_core = elems/ncores;
            if(chunk_per_core == 0){
                printf("Chunk for each core is 0!\n");
                return -2;
            }
            // Offset to index the correct chunk of data per core
            uint32_t offset = core_idx*chunk_per_core;
            
            matrix_trans_opt(chunk_per_core, offset, &start_cycle[core_idx], &end_cycle[core_idx],
                            src, dst);

            // Performance calculations for each core
            total_cycles[core_idx] = end_cycle[core_idx]-start_cycle[core_idx];
            flop_cycle[core_idx] = (double) (chunk_per_core*elems)/ (double) total_cycles[core_idx];
        }
    }


    snrt_cluster_hw_barrier(); // Barrier syncronization
            
    if(core_idx==0){

        // Mean performance values
        uint64_t mean_cycles=0;
        double mean_flop_cycle = 0.0;

        for(uint32_t cid = 0; cid < ncores; cid ++){
            mean_cycles += total_cycles[cid];
            mean_flop_cycle += flop_cycle[cid];
        }
        mean_cycles /= ncores;
        mean_flop_cycle /= ncores;

        if(single_core) printf("Single core benchmark\n");

        printf("Matrix transpose %dx%d performance\n",elems,elems);
        printf("Mean cycles: %llu\n", (unsigned long long)mean_cycles);
        printf("Mean FLOP/cycle: %f\n", mean_flop_cycle);

        // Print all results, not recommended for matrices larger than 8x8
        if(elems==8){
            printf("         0    1    2    3    4    5    6    7 \n");
            for(uint32_t i=0; i<elems; i++){
                printf("Row %d: ", i);
                for(uint32_t j=0; j<elems;j++){

                    printf("%2.1f; ",dst[i*elems + j]);

                }
                printf("\n");
            }
        }   // Print results for sanity check
        else if(PRINT_RESULTS){
            for(uint32_t i=0; i<7; i++){ // first eight elements of first row
                printf("MAT_trans(0,%d): %f\n",i, dst[i]);
            }
            // first eight elements of last row
            for(uint32_t i=0; i<7; i++){
                printf("MAT_trans(%d, %d): %f\n",elems-1,i, dst[elems*(elems-1)+i]);
            }
        }
    }
    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */