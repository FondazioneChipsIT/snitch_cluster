// Luca Colombo Chips-IT 2026
// SVM classification using RBF kernel, using the BILL dataset.
// No use of SSRs or frep, the BILL dataset has only 4 features, so the overhead of setting up the SSR outweighs the benefits.
#include "snrt.h"
#include "data.h"
#include "svm_rbf.h"

uint32_t CHECK_RESULTS = 1; // Set to 1 to check results against golden output

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();
    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        data_model_TCDM = (float *)snrt_l1_next();
        pred_TCDM = data_model_TCDM + L * F_DIM;
        x_ref_TCDM = pred_TCDM + L;
        sv_coef_TCDM = x_ref_TCDM + COEF_DIM * F_DIM;

        size_t size = L * F_DIM * sizeof(float);
        size_t size_sv = COEF_DIM * F_DIM * sizeof(float);

        snrt_dma_start_1d(data_model_TCDM, data_model, size);
        snrt_dma_start_1d(x_ref_TCDM, sv, size_sv);
        snrt_dma_start_1d(sv_coef_TCDM, sv_coef, COEF_DIM * sizeof(float));
        snrt_dma_wait_all();

    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        uint32_t chunk_per_core = L / ncores;
        uint32_t offset = core_idx * chunk_per_core;
        
        SVM_RBF(core_idx, chunk_per_core, offset, 
                data_model_TCDM, pred_TCDM, x_ref_TCDM, bias, sv_coef_TCDM, GAMMA, F_DIM);
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        for(uint32_t i = 0; i < L; i++){
            if(fabsf(pred_TCDM[i] - golden[i]) > 0.1f){ // Using a tolerance of 0.1 for classification output
                err ++;
            }
        }
    }

    return err;
}