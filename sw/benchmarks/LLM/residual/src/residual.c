// Luca Colombo Chips-IT 2025
/* Residual that adds original to result of CONV3x3 */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "residual_opt.h"

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();
    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        x = (float *)snrt_l1_next();
        out = x + M*N;
        y = out + M*N;

        size_t size = M * N * sizeof(float);
        snrt_dma_start_1d(x, mat_a, size);
        snrt_dma_start_1d(y, mat_b, size);
        snrt_dma_wait_all();

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization
    snrt_mcycle();
    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the matrix, represents
        // the number of rows each core has to use
        uint32_t chunk_per_core = M/ncores;
        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx*chunk_per_core;

        residual_opt(core_idx,chunk_per_core, offset, x, y, out);

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization
    snrt_mcycle();
    
    uint32_t err = 0;
    float eps = 1e-5f;

    if (core_idx == 0) {
        for(uint32_t i = 0; i < M * N; i++){
            if(fabsf(out[i] - golden[i]) > eps){
                err ++;
            }
        }
        printf("Errors: %u\n", err);
    }

    return err;
}
