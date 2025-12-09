// Luca Colombo 2025 CHIPS-IT
/* Matrix multiplication dst = A*trans(B), only with square matrices */
#include "snrt.h"
#include "data.h"
#include "matrix_trans_opt.h"
#include "matrix_mul_opt.h"

/* Print first/last elements  */
bool PRINT_RESULTS = 1;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* DM core allocates and initializes matrices in TCDM */
    if (snrt_is_dm_core()) {


        mat_a = (double *)snrt_l1_next();
        mat_b = mat_a + elems * elems;
        trans_b = mat_b + elems * elems;
        dst = trans_b + elems * elems;

        if (!mat_a || !mat_b || !trans_b || !dst) {
            printf("Memory allocation failed!\n");
            return -1;
        }

        /* deterministic init */
        for (uint32_t i = 0; i < elems * elems; ++i) mat_a[i] = (double)i;
        for (uint32_t i = 0; i < elems * elems; ++i) mat_b[i] = (double)i;
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
         // Transpose of mat_a (not considered in performance)
        matrix_trans_opt(chunk_per_core, offset, mat_b, trans_b); 
        // Call the kernel
        matrix_mul_opt(chunk_per_core,offset, &start_cycle[core_idx], &end_cycle[core_idx], mat_a, trans_b, dst);

        total_cycles[core_idx] = end_cycle[core_idx] - start_cycle[core_idx];
        //                      2 fmadd per elems(row) *elems (in a row) + 1 fadd each elems time
        flop_cycle[core_idx] = ( chunk_per_core * elems *elems * 2.0 + (double)chunk_per_core * elems) / (double)total_cycles[core_idx];
    }

    snrt_cluster_hw_barrier();

    if (core_idx == 0) {

        // Mean performance values
        uint64_t mean_cycles=0;
        double mean_flop_cycle = 0.0;
        double total_flop_cycle = 0.0;

        for(uint32_t cid = 0; cid < ncores; cid ++){
            mean_cycles += total_cycles[cid];
            total_flop_cycle += flop_cycle[cid];
        }
        mean_cycles /= ncores;
        mean_flop_cycle = total_flop_cycle/ncores;

        printf("Matrix multiplication transpose B %dx%d performance\n",elems,elems);
        printf("Mean cycles: %llu\n", (unsigned long long)mean_cycles);
        printf("Mean FLOP/cycle: %f\n", mean_flop_cycle);
        printf("Total FLOP/cycle: %f\n", total_flop_cycle);

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
        else if(PRINT_RESULTS){
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