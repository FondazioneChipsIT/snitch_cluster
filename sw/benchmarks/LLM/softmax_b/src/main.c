// 2026 Luca Colombo Chips-IT

#include "data.h"

#include "softmax.h"

#include "snrt.h"

// Pointers to TCDM
float *ifmap_TCDM, *ofmap_TCDM;

uint32_t CHECK_RESULTS = 1;

int main() {

    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    uint32_t total_elems = BATCH_SIZE * SEQ_LEN * INPUT_SAMPLES;

    /* DM core allocates and initializes elements in TCDM */
    if (snrt_is_dm_core()) {

        ifmap_TCDM = (float *)snrt_l1_next();
        ofmap_TCDM = ifmap_TCDM + total_elems;

        size_t size = total_elems * sizeof(float);

        snrt_dma_start_1d(ifmap_TCDM, ifmap, size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // determine the row offset for each core
        uint32_t row_offset = core_idx * INPUT_SAMPLES;

        // determine the row stride of each matrix
        uint32_t row_stride = ncores * INPUT_SAMPLES;

        // determine the batch offset for each core
        uint32_t batch_offset = SEQ_LEN * INPUT_SAMPLES;

        // chunck per core
        uint32_t chunk_size = SEQ_LEN / ncores;


        softmax_FP32(&ifmap_TCDM[row_offset], &ofmap_TCDM[row_offset], 
                    row_stride, batch_offset,  BATCH_SIZE, chunk_size, INPUT_SAMPLES);

    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();
    
    uint32_t err = 0;
    float eps = 1e-5f;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        for(uint32_t i = 0; i < total_elems; i++){
            if(fabsf(ofmap_TCDM[i] - golden[i]) > eps){
                err ++;
            }
        }
    }

    return err; 
}
