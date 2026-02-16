// Luca Colombo 2025 CHIPS-IT
/* Matrix multiplication dst_TCDM = A*B, only with square matrices */
#include "snrt.h"
#include "data.h"
#include "matrix_mul_opt.h"

/* Print first/last elements  */
uint32_t CHECK_RESULTS = 1;

uint32_t use_opt = 1;

void matmul_simple_f32(uint32_t chunk_per_core, uint32_t offset,
                    float *mat_a_TCDM, float* mat_b_TCDM, float *dst_TCDM){
    
    snrt_mcycle();

    for (int i = offset; i < offset + chunk_per_core; ++i) {
        for (int j = 0; j < elems; ++j) {
            float acc = 0.0;

            for (int k = 0; k < elems; ++k) {
                acc += mat_a_TCDM[i * elems + k] * mat_b_TCDM[k * elems + j];
            }

            dst_TCDM[i * elems + j] = acc;
        }
    }

    snrt_mcycle();

    return;
}


int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* DM core allocates and initializes matrices in TCDM */
    if (snrt_is_dm_core()) {


        mat_a_TCDM = (float *)snrt_l1_next();
        mat_b_TCDM = mat_a_TCDM + elems * elems;
        dst_TCDM = mat_b_TCDM + elems * elems;

        if (!mat_a_TCDM || !mat_b_TCDM || !dst_TCDM) {
            printf("Memory allocation failed!\n");
            return -1;
        }

        size_t size = elems * elems * sizeof(float);

        snrt_dma_start_1d(mat_a_TCDM, mat_a, size);
        snrt_dma_start_1d(mat_b_TCDM, mat_b, size);
        snrt_dma_start_1d(dst_TCDM, golden, size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        /* calculate chunks per core */
        uint32_t chunk_per_core = elems / ncores;

        /* compute offset for this core */
        uint32_t offset = core_idx * chunk_per_core;

        if (chunk_per_core == 0) {
            if (core_idx == 0) printf("ERROR: chunk_per_core == 0 (increase matrix size or reduce ncores)\n");
            return 0;
        }

        if(use_opt==1) 
            matrix_mul_opt(chunk_per_core,offset, mat_a_TCDM, mat_b_TCDM, dst_TCDM);
        else
            matmul_simple_f32(chunk_per_core,offset, mat_a_TCDM, mat_b_TCDM, dst_TCDM);

    }

    snrt_cluster_hw_barrier();

    uint32_t err = 0;
    float eps = 1e-5;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        asm("nop \n");
        for(uint32_t i = 0; i < elems * elems; i++){
            if(fabs(dst_TCDM[i] - golden[i]) > eps){
                err ++;
            }
        }
        printf("Errors: %u\n", err);
    }

    return err;
}