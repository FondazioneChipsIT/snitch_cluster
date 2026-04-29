// Luca Colombo Chips-IT 2025
/* CONV3x3 2d filter with no padding */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "conv3x3_opt.h"
#include "conv3x3_naive.h"

uint32_t CHECK_RESULTS = 1; // Set to 1 to check results against golden output

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();
    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        input_TCDM = (float *)snrt_l1_next();
        dst_TCDM = input_TCDM + FM_ROWS * FM_ROWS;
        kernel_TCDM = dst_TCDM + OUT_ROWS * OUT_ROWS;

        size_t size = FM_ROWS * FM_ROWS * sizeof(float);
        size_t size_filter = 3 * 3 * sizeof(float);

        snrt_dma_start_1d(input_TCDM, input_fm, size);
        snrt_dma_start_1d(kernel_TCDM, conv_kernel, size_filter);
        snrt_dma_wait_all();

    }

    // Compute the chunk of the matrix, represents
    // the number of rows each core has to use
    uint32_t chunk_per_core = FM_ROWS/ncores;
    // Offset to index the correct chunk of data per core
    uint32_t offset = core_idx*chunk_per_core;
        
    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Only the compute cores do something
    if(snrt_is_compute_core()){
        // Call the kernel   
        conv3x3_opt(core_idx,chunk_per_core, offset, input_TCDM, dst_TCDM, kernel_TCDM);
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        for(uint32_t i = 0; i < OUT_ROWS * OUT_ROWS; i++){
            if(fabsf(dst_TCDM[i] - golden[i]) > 1e-5f){
                err ++;
            }
        }
    }

    return err;
}