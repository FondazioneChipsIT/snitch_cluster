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

int16_t *x_TCDM;
// Partial votes of each core, reduced inside rf_predict_int_model.
// In TCDM, like the barrier: they cross a barrier between cores.
int *rf_votes;
// The barrier must NOT be a plain global: globals are linked into L3 (the
// linker script only has the DRAM region), so every snrt_partial_barrier would
// spin on a DRAM address at ~60 cycles per access. It is allocated in TCDM by
// the DM core in main instead, and this global only holds the pointer.
snrt_barrier_t *barr;

#include "random_forest.h"

uint32_t CHECK_RESULTS = 1;

int main(){
    uint32_t NUM_CORES = snrt_cluster_compute_core_num();
    uint32_t core_idx = snrt_cluster_core_idx();
    int pred = -1;

    if(snrt_is_dm_core()){
        // Barrier in TCDM: as a global it would be linked into DRAM and every
        // snrt_partial_barrier would spin on it at ~60 cycles per access.
        // Allocated first, so the layout below is placed after it.
        barr = (snrt_barrier_t *)snrt_l1_alloc(sizeof(snrt_barrier_t));
        barr->cnt = 0;
        barr->iteration = 0;

        rf_votes = (int *)snrt_l1_alloc(NUM_CORES * RF_N_CLASSES * sizeof(int));

       
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

        // NOT snrt_l1_next(): the allocator never advanced, so that returns
        // the same address as RF_TREE_0 and the features would overwrite the
        // first tree. Place it after the last tree instead.
        x_TCDM = (int16_t *)(RF_TREE_5 + RF_TREE_5_SIZE);
        size_t x_size = RF_N_FEATURES* sizeof(int16_t);
        snrt_dma_start_1d(x_TCDM, (void *)x_dram, x_size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Only the compute cores do something
    if(snrt_is_compute_core()){
        snrt_barrier_t *bar_p = barr;   // local copy, see note above

        for (int i = 0; i < 10; i++){
            pred = rf_predict_int_model(core_idx, NUM_CORES, x_TCDM, i);
            snrt_partial_barrier(bar_p, 8);
        }

    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        // Reference: majority vote over ALL the trees. Walked from the copies
        // in DRAM, which are the only intact ones (x_TCDM is allocated at the
        // same address as RF_TREE_0, so the TCDM copy of tree 0 is overwritten
        // by the feature vector).
        const DTNode *ref_trees[RF_N_TREES] = {
            RF_TREE_0_dram, RF_TREE_1_dram, RF_TREE_2_dram,
            RF_TREE_3_dram, RF_TREE_4_dram, RF_TREE_5_dram
        };

        int votes[RF_N_CLASSES] = {0};
        for (int t = 0; t < RF_N_TREES; t++) {
            const DTNode *T = ref_trees[t];
            int id = 0;
            for (;;) {
                const DTNode *n = &T[id];
                if (n->feature < 0) { votes[(unsigned)n->leaf_class]++; break; }
                id = (x_dram[n->feature] <= n->threshold) ? n->left : n->right;
            }
        }

        int ref = 0;
        for (int c = 1; c < RF_N_CLASSES; c++)
            if (votes[c] > votes[ref]) ref = c;

        if (pred != ref) {
            err ++;
        }
        printf("Errors: %u (pred %d, ref %d)\n", err, pred, ref);
    }

    return err;
}