// Luca Colombo Chips-IT 2025
/* LU optimized multicore version with SSR and FREP */

#include "snrt.h"
#include "data.h"
#include "verify_lu.h"
#include "lu_decomp_opt.h"
//#include "lu_decomp_naive.h"

// Use optimized or naive version
uint32_t use_opt = 1;

// Verify results, slow and warnings for sqrt
uint32_t verify_results = 0;


// In a separate noinline function on purpose: this check needs enough live FP
// values that the compiler spills fs0/fs1, and an fsd in the prologue of main()
// is executed by the DM core too, which has no D support and traps on it.
static __attribute__((noinline)) uint32_t check_lu(void) {
    uint32_t err = 0;
    float eps = 1e-3f;

    // L*U must reproduce the permuted original matrix, P*A.
    // L is unit lower triangular and U upper triangular, both packed in mat.
    // Done element by element to avoid the big temporary buffers of
    // verify_lu(), which would not fit on the stack.
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            uint32_t kmax = (i < j) ? i : j;
            float lu = 0.0f;

            for (uint32_t k = 0; k <= kmax; k++) {
                float l = (k < i) ? mat[i*cols + k] : 1.0f;
                lu += l * mat[k*cols + j];
            }

            float pa = orig_buf[perm_vec[i]*cols + j];
            // relative tolerance: the entries grow up to ~100
            if (fabsf(lu - pa) > eps * (1.0f + fabsf(pa))) {
                err ++;
            }
        }
    }
    printf("Errors: %u\n", err);
    return err;
}

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    /* Allocate on DM core*/
    if (snrt_is_dm_core()) {
        // Barrier in TCDM: as a global it would be linked into DRAM and every
        // snrt_partial_barrier would spin on it at ~60 cycles per access.
        // Allocated first, so the layout below is placed after it.
        barr = (snrt_barrier_t *)snrt_l1_alloc(sizeof(snrt_barrier_t));
        barr->cnt = 0;
        barr->iteration = 0;


        mat = (float *)snrt_l1_next();
        orig_buf = mat + rows * cols;
        perm_vec = (int *)(orig_buf + rows * cols);
        // cols*ncore
        vec_write_back = (float *)(perm_vec + rows);
        row_k = vec_write_back + cols*ncores;
        row_b = row_k + cols;

        if (!mat || !orig_buf || !perm_vec || !vec_write_back) {
            printf("Memory allocation failed!\n");
            return -1;
        }

        for (uint32_t i=0; i<rows; i++) {
            for (uint32_t j=0; j<cols; j++) {
                mat[i*cols + j] = ((i + j) % cols) + 1 + i;
                orig_buf[i*cols + j] = mat[i*cols + j];
            }
        }

        for (uint32_t i=0;i<rows;i++) perm_vec[i] = i;
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // kernel call
    if(snrt_is_compute_core()) {
        //if(use_opt == 1)
        lu_decomp_opt(core_idx, ncores, &start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec,
                        row_k, row_b, vec_write_back);
        /*else
            lu_decomp_naive(core_idx, ncores, &start_cycle[core_idx], &end_cycle[core_idx], mat, perm_vec);*/
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;

    if (verify_results == 1 && core_idx == 0) {
        err = check_lu();
    }

    return err;
}
