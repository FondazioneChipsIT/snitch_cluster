// Luca Colombo 2026 CHIPS-IT
/* GEMM with C = alpha* A*B + beta*C */
#include "snrt.h"
#include "data.h"
#include "gemm_fp32.h"

/* Print first/last elements  */
uint32_t CHECK_RESULTS = 1;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* DM core allocates and initializes matrices in TCDM */
    if (snrt_is_dm_core()) {

        // Load all matrices in TCDM
        mat_a_TCDM = (float *)snrt_l1_next();
        mat_b_TCDM = mat_a_TCDM + M * K;
        mat_c_TCDM = mat_b_TCDM + K * N;

        size_t size_a = M * K * sizeof(float);
        size_t size_b = K    * N * sizeof(float);
        size_t size_c = M * N * sizeof(float);

        snrt_dma_start_1d(mat_a_TCDM, mat_a, size_a);
        snrt_dma_start_1d(mat_b_TCDM, mat_b, size_b);
        snrt_dma_start_1d(mat_c_TCDM, mat_c, size_c);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        /* calculate chunks per core, split the columns of the b matrix */
        uint32_t chunk_per_core = N / ncores;

        /* compute offset for this core */
        uint32_t offset = core_idx * chunk_per_core;
        uint32_t m = M;
        uint32_t n = N;
        uint32_t k = K;
        
        gemm_fp32(chunk_per_core,offset, alpha, beta, mat_a_TCDM, mat_b_TCDM, mat_c_TCDM, m, n, k);

    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;
    float eps = 5e-5f;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        asm("nop \n");
        for(uint32_t i = 0; i < M * N; i++){
            if(fabsf(mat_c_TCDM[i] - golden[i]) > eps){
                err ++;
            }
        }
        printf("Errors: %u\n", err);
    }

    return err;
}