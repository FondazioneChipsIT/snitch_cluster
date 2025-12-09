// Luca Colombo Chips-IT 2025
/* Sets all elements of mat_a to val */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "matrix_set_all_opt.h"

// Print first 5 and last 5 results
bool PRINT_RESULTS = 1;

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        mat_a = (double *)snrt_l1_next();
        val = mat_a + rows*col; // The matrix is put in L1 by rows, so we need to allocate
        // rows*col elements

        // If pointers are null -> break
        if (!mat_a || !val) {
            printf("Memory allocation failed!\n");
            return -1;
        }
        
        *val = 9.999;
    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the vector per core, and the offset that is used to space them
        uint32_t chunk_per_core = (rows*col)/ncores;
        if(chunk_per_core == 0){
            printf("Chunk for each core is 0!\n");
            return -2;
        }
        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx*chunk_per_core;
   
        matrix_set_all_opt(chunk_per_core, offset, &start_cycle[core_idx], &end_cycle[core_idx],
                          mat_a, val);

        // Performance calculations for each core
        total_cycles[core_idx] = end_cycle[core_idx]-start_cycle[core_idx];
        flop_cycle[core_idx] = (double) chunk_per_core/ (double) total_cycles[core_idx];
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

        printf("Matrix set all %dx%d performance\n",rows,col);
        printf("Mean cycles: %llu\n", (unsigned long long)mean_cycles);
        printf("Mean FLOP/cycle: %f\n", mean_flop_cycle);

        /* for(uint32_t i=0; i<rows; i++){
            for(uint32_t j=0; j<col;j++){

                printf("mat_a(%d,%d): %f",i,j,mat_a[i*j]);

            }
            printf("\n");
        }*/

        // Print results for sanity check
        if(PRINT_RESULTS){
            for(uint32_t i=0; i<5; i++){ // first five elements of first row
                printf("MAT_set(0,%d): %f\n",i, mat_a[i]);
            }
            // first five elements of last row
            for(uint32_t i=0; i<5; i++){
                printf("MAT_set(%d, %d): %f\n",rows,i, mat_a[rows*(col-1)+i]);
            }
        }
    }
    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */