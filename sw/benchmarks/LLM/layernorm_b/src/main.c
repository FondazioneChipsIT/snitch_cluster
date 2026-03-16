// 2026 Luca Colombo Chips-IT

#include "data.h"

#include "layernorm_2.h"

#include "snrt.h"

int main() {

    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    uint32_t total_elems = BATCH_SIZE * SEQ_LEN * EMBEDDINGS;

    /* DM core allocates and initializes elements in TCDM */
    if (snrt_is_dm_core()) {

        ifmap_TCDM = (float *)snrt_l1_next();
        ofmap_TCDM = ifmap_TCDM + total_elems;
        size_t size = total_elems * sizeof(float);

        snrt_dma_start_1d(ifmap_TCDM, input, size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Divide the work among the cores
        uint32_t rows_per_core = (BATCH_SIZE * SEQ_LEN) / ncores;

        uint32_t offset = core_idx * rows_per_core * EMBEDDINGS;

        layernorm(ifmap_TCDM + offset, ofmap_TCDM + offset, rows_per_core);

    }

    snrt_cluster_hw_barrier();

    uint32_t err = 0;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        for(uint32_t i = 0; i < total_elems; i++){
            if(fabsf(ofmap_TCDM[i] - O_golden[i]) > (float) EPS){
                err ++;
            }
        }
    }

    return err; 
}
