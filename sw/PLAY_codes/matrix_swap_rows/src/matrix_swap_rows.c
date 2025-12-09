// Luca Colombo Chips-IT 2025
/* Swap row1 and row2 */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "matrix_swap_rows_opt.h"

// Print first 5 and last 5 results
bool PRINT_RESULTS = 1;

// Naive version for performance comparison (not easy)
bool USE_OPT = 1;
void matrix_swap_rows_naive(uint32_t chunk_per_core, uint32_t offset, 
                        uint64_t *start_cycle, uint64_t *end_cycle, 
                        double *mat_a, double *row_a, double *row_b,
                        uint32_t indx_a, uint32_t indx_b){

    *start_cycle = snrt_mcycle();
    // Copy rows in buffers
    for(uint32_t i = offset; i<offset+chunk_per_core; i++){
        row_a[i] = mat_a[indx_a*col + i ];
        row_b[i] = mat_a[indx_b*col + i ];
    }
    // Copy row_a in mat_a(indxb) and viceversa
    for(uint32_t i = offset; i<offset+chunk_per_core; i++){
        mat_a[indx_a*col + i ] = row_b[i];
        mat_a[indx_b*col + i ] = row_a[i];
    }

    *end_cycle = snrt_mcycle();

    return;
}

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        mat_a = (double *)snrt_l1_next();
        row_a = mat_a + rows*col; // The matrix is put in L1 by rows, so we need to allocate
        // rows*col elements
        row_b = row_a + col;

        // If pointers are null -> break
        if (!mat_a || !row_a || !row_b) {
            printf("Memory allocation failed!\n");
            return -1;
        }

        for(int i = 0; i < rows*col; i++){
            mat_a[i] = (double)i;   
        }
        
    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the rows (col elements) per core
        uint32_t chunk_per_core = col/ncores;
        if(chunk_per_core == 0){
            printf("Chunk for each core is 0!\n");
            return -2;
        }
        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx*chunk_per_core;

        if(USE_OPT){
            matrix_swap_rows_opt(chunk_per_core, offset, &start_cycle[core_idx], &end_cycle[core_idx],
                            mat_a, row_a, row_b, indx_a, indx_b);
        }else{
            matrix_swap_rows_naive(chunk_per_core, offset, &start_cycle[core_idx], &end_cycle[core_idx],
                            mat_a, row_a, row_b, indx_a, indx_b);
        }

        // Performance calculations for each core
        total_cycles[core_idx] = end_cycle[core_idx]-start_cycle[core_idx];

        // If using SSR and FREP 4 FLOP per elements (fadd)
        if(USE_OPT){
            flop_cycle[core_idx] = (double) 4*chunk_per_core/ (double) total_cycles[core_idx];
        }
        else{
            flop_cycle[core_idx] = (double) chunk_per_core/ (double) total_cycles[core_idx];
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

        printf("Matrix swap rows %dx%d performance\n",rows,col);
        printf("Mean cycles: %llu\n", (unsigned long long)mean_cycles);
        printf("Mean FLOP/cycle: %f\n", mean_flop_cycle);

        // Print results for sanity check, first 4 elements of each row for speed
        if(PRINT_RESULTS){
            printf("Row %d:", indx_a);
            for(uint32_t i=0; i<4; i++){
                printf(" %f ", mat_a[indx_a*col+i]);
            }
            printf("...");
            for(uint32_t i=col-4; i<col; i++){
                printf(" %f ", mat_a[indx_a*col+i]);
            }
            printf("\n");

            printf("Row %d:", indx_b);
            for(uint32_t i=0; i<4; i++){
                printf(" %f ", mat_a[indx_b*col+i]);
            }
            printf("...");
            for(uint32_t i=col-4; i<col; i++){
                printf(" %f ", mat_a[indx_b*col+i]);
            }
            printf("\n");
        }
    }
    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */