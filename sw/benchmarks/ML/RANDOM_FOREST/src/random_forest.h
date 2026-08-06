#include "snrt.h"

static const uint16_t RF_TREES_SIZES[RF_N_TREES] = { RF_TREE_0_SIZE, RF_TREE_1_SIZE, RF_TREE_2_SIZE, RF_TREE_3_SIZE, RF_TREE_4_SIZE, RF_TREE_5_SIZE };

static inline int rf_predict_int_model(uint32_t core_idx, uint32_t NUM_CORES, const int16_t x[RF_N_FEATURES], int i){
  DTNode* RF_TREES[RF_N_TREES];
  RF_TREES[0] = RF_TREE_0;
  RF_TREES[1] = RF_TREE_1;
  RF_TREES[2] = RF_TREE_2;
  RF_TREES[3] = RF_TREE_3;
  RF_TREES[4] = RF_TREE_4;
  RF_TREES[5] = RF_TREE_5;

  int blockSize = (RF_N_TREES + NUM_CORES-1)/NUM_CORES;
  int start = core_idx*blockSize;
  int end = start + blockSize < RF_N_TREES? start + blockSize : RF_N_TREES;

  int votes[RF_N_CLASSES]={0};

  for (int t=start;t<end;t++){
    const DTNode* T = RF_TREES[t];
    int id=0;
    for(;;){
      const DTNode* n=&T[id];

      if(n->feature<0){
        votes[(unsigned)n->leaf_class]++;
        break;
      }
      id = (x[n->feature] <= n->threshold) ? n->left : n->right;
    }
  }

  // Each core only walked its own slice of the trees, so these are PARTIAL
  // votes: they have to be reduced across the cores, otherwise no core ever
  // computes the prediction of the whole forest.
  for (int c=0;c<RF_N_CLASSES;c++)
    rf_votes[core_idx*RF_N_CLASSES + c] = votes[c];

  snrt_partial_barrier(barr, 8);

  for (int c=0;c<RF_N_CLASSES;c++){
    int tot=0;
    for (uint32_t p=0;p<NUM_CORES;p++)
      tot += rf_votes[p*RF_N_CLASSES + c];
    votes[c]=tot;
  }

  // best must start at class 0: it used to start at the loop index i coming
  // from main, which for i >= RF_N_CLASSES read votes[] out of bounds
  int best=0;

  for (int c=1;c<RF_N_CLASSES;c++)
    if (votes[c]>votes[best]) best=c;

  return best;
}
