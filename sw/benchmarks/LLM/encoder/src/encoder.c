// Luca Colombo Chips-IT 2025

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "encoder_opt.h"

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();
    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){
        wte_TCDM = (float *)snrt_l1_next();
        wpe_TCDM = wte_TCDM + V_USED * C;
        out_TCDM = wpe_TCDM + T * C;

        snrt_dma_start_1d(wte_TCDM, wte_compact, V_USED*C * sizeof(float));
        snrt_dma_start_1d(wpe_TCDM, wpe, T*C  * sizeof(float));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier(); // Barrier syncronization
    snrt_mcycle();
    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the matrix, represents
        // the number of rows each core has to use
        uint32_t chunk_per_core = B*T/ncores;
        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx*chunk_per_core;

        encoder_opt(chunk_per_core, offset, inp, wte_TCDM, wpe_TCDM, out_TCDM);

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization
    snrt_mcycle();
    
    uint32_t err = 0;
    float eps = 1e-5f;

    // CHECK RESULTS

    if (core_idx == 0) {
        for(uint32_t i = 0; i < B * T * C; i++){
            if(fabsf(out_TCDM[i] - golden[i]) > eps){
                err ++;
            }
        }
        printf("Errors: %u\n", err);
    }

    return err;
}
