// Luca Colombo 2025 CHIPS-IT
/* Matrix multiplication dst = A*B, only with square matrices */
#include "snrt.h"
#include "data.h"
#include "matrix_mul_opt.h"

/* Print first/last elements  */
uint32_t PRINT_RESULTS = 0;

uint32_t use_opt = 1;

void matmul_simple_f32(uint32_t chunk_per_core, uint32_t offset,
                    float *mat_a, float* mat_b, float *dst){
    
    snrt_mcycle();

    for (int i = offset; i < offset + chunk_per_core; ++i) {
        for (int j = 0; j < elems; ++j) {
            float acc = 0.0;

            for (int k = 0; k < elems; ++k) {
                acc += mat_a[i * elems + k] * mat_b[k * elems + j];
            }

            dst[i * elems + j] = acc;
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


        mat_a = (float *)snrt_l1_next();
        mat_b = mat_a + elems * elems;
        dst = mat_b + elems * elems;

        if (!mat_a || !mat_b || !dst) {
            printf("Memory allocation failed!\n");
            return -1;
        }

        /* deterministic init */
        for (uint32_t i = 0; i < elems * elems; ++i) mat_a[i] = (float)i;
        for (uint32_t i = 0; i < elems * elems; ++i) mat_b[i] = (float)i;
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
            matrix_mul_opt(chunk_per_core,offset, mat_a, mat_b, dst);
        else
            matmul_simple_f32(chunk_per_core,offset, mat_a, mat_b, dst);

    }

    snrt_cluster_hw_barrier();

    if (PRINT_RESULTS == 1 && core_idx == 0) {

       // Print all results, not recommended for matrices larger than 8x8
        if(elems == 8){

            for(uint32_t i=0; i<elems; i++){
                printf("Row %d: ", i);
                for(uint32_t j=0; j<elems;j++){

                    printf("%.1f; ",dst[i*elems + j]);

                }
                printf("\n");
            }
        }   // Print results for sanity check
        else{
            for(uint32_t i=0; i<7; i++){ // first eight elements of first row
                printf("mat_mul(0,%d): %.2f\n",i, dst[i]);
            }
            // first eight elements of last row
            for(uint32_t i=0; i<7; i++){
                printf("mat_mul(%d, %d): %.2f\n",elems-1,i, dst[elems*(elems-1)+i]);
            }
        }
    }

    return 0;
}