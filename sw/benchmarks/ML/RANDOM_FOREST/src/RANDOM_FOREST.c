#include <stdint.h>
#include "snrt.h"

typedef struct { int8_t feature; int16_t threshold; int16_t left; int16_t right; int8_t leaf_class; int8_t padding[6];} DTNode;

DTNode *RF_TREE_0;
DTNode *RF_TREE_1;
DTNode *RF_TREE_2;
DTNode *RF_TREE_3;
DTNode *RF_TREE_4;
DTNode *RF_TREE_5;

#include "data.h"
#include "random_forest.h"

int16_t *x_TCDM;
snrt_barrier_t barr;

int main(){
    uint32_t NUM_CORES = snrt_cluster_compute_core_num();
    uint32_t core_idx = snrt_cluster_core_idx();

    if(snrt_is_dm_core()){
       
        RF_TREE_0 = (DTNode *)snrt_l1_next();
        RF_TREE_1 = RF_TREE_0 + RF_TREE_0_SIZE;
        RF_TREE_2 = RF_TREE_1 + RF_TREE_1_SIZE;         
        RF_TREE_3 = RF_TREE_2 + RF_TREE_2_SIZE;
        RF_TREE_4 = RF_TREE_3 + RF_TREE_3_SIZE;
        RF_TREE_5 = RF_TREE_4 + RF_TREE_4_SIZE;

        size_t size[6] = {RF_TREE_0_SIZE * sizeof(DTNode), RF_TREE_1_SIZE * sizeof(DTNode), RF_TREE_2_SIZE * sizeof(DTNode), RF_TREE_3_SIZE * sizeof(DTNode), RF_TREE_4_SIZE * sizeof(DTNode), RF_TREE_5_SIZE * sizeof(DTNode)};

        snrt_dma_start_1d(RF_TREE_0, (void *)RF_TREE_0_dram, size[0]);
        snrt_dma_start_1d(RF_TREE_1, (void *)RF_TREE_1_dram, size[1]);
        snrt_dma_start_1d(RF_TREE_2, (void *)RF_TREE_2_dram, size[2]);
        snrt_dma_start_1d(RF_TREE_3, (void *)RF_TREE_3_dram, size[3]);
        snrt_dma_start_1d(RF_TREE_4, (void *)RF_TREE_4_dram, size[4]);
        snrt_dma_start_1d(RF_TREE_5, (void *)RF_TREE_5_dram, size[5]);

        snrt_dma_wait_all();

        x_TCDM = (int16_t *)snrt_l1_next();
        size_t x_size = RF_N_FEATURES* sizeof(int16_t);
        snrt_dma_start_1d(x_TCDM, (void *)x_dram, x_size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        for (int i = 0; i < 10; i++){
            rf_predict_int_model(core_idx, NUM_CORES, x_TCDM, i);
            snrt_partial_barrier(&barr, 8);
        }

    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    return 0;
}