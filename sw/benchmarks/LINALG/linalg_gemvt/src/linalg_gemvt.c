// Luca Colombo 2026 CHIPS-IT
/* Gemvt */
#include "snrt.h"
#include "data.h"
#include "gemvt_opt.h"

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* DM core allocates and initializes matrices in TCDM */
    if (snrt_is_dm_core()) {

        mat_a_TCDM = (float *)snrt_l1_next();
        vec_x_TCDM = mat_a_TCDM + M * N;
        dst_TCDM = vec_x_TCDM + M;

        size_t size_mat = M * N * sizeof(float);
        size_t size_vec = M * sizeof(float);

        snrt_dma_start_1d(mat_a_TCDM, mat_a, size_mat);
        snrt_dma_start_1d(vec_x_TCDM, vec_x, size_vec);
        snrt_dma_wait_all();
    }

    uint32_t m = M;
    uint32_t n = N;
    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        /* calculate chunks per core, divide columns */
        uint32_t chunk_per_core = N / ncores;

        /* compute offset for this core */
        uint32_t offset = core_idx * chunk_per_core;

        gemvt_opt(chunk_per_core, offset, mat_a_TCDM, vec_x_TCDM, alpha, dst_TCDM, m, n);

    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;
    float eps = 1e-4f;
s
    if (core_idx == 0) {
        asm("nop \n");
        for(uint32_t i = 0; i < N; i++){
            if(fabsf(dst_TCDM[i] - golden[i]) > eps){
                err ++;
            }
        }
    }

    return err;
}