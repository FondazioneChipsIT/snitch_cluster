#include <math.h>
#include "snrt.h"

snrt_barrier_t barr;

uint32_t svd_jacobi_opt(float *mat, float *mat_V, float *vec_S, const uint32_t dim_M){
    
    uint32_t pairs_per_round;
    uint32_t even_dim;
    uint32_t iter;
    uint32_t id;
    float max_offdiag;

    id = snrt_cluster_core_idx();
    uint32_t NUM_CORES = snrt_cluster_compute_core_num();
    even_dim = (dim_M % 2 == 0) ? dim_M : (dim_M + 1);
    pairs_per_round = even_dim / 2;

    iter = 0;
    while (iter++ < MAX_ITER) {

        for (uint32_t round = 0; round < (even_dim - 1); round++) {
            uint32_t pair_start;
            uint32_t pair_end;
            uint32_t block;
            uint32_t left;

            block = pairs_per_round / NUM_CORES;
            left = pairs_per_round % NUM_CORES;
            pair_start = id * block + (id < left ? id : left);
            pair_end = pair_start + block + (id < left ? 1 : 0);

            local_max[id] = 0;

            if (pair_start >= pair_end) {
                return 1;
            }

            for (uint32_t pair = pair_start; pair < pair_end; pair++) {
                bool compute;
                float cos;
                float sin;
                float tau;
                float t;
                uint32_t i;
                uint32_t j;

                snrt_partial_barrier(&barr, 8);

                i = (round + pair) % (even_dim - 1);
                j = (pair == 0) ? (even_dim - 1) : ((even_dim - 1 - pair + round) % (even_dim - 1));

                compute = true;
                if (i >= dim_M || j >= dim_M)
                    compute = false;

                if (fabs(mat[i * dim_M + j]) < EPSILON)
                    compute = false;

                if (compute) {
                    tau = (mat[j * dim_M + j] - mat[i * dim_M + i]) / (TWO_f * mat[i * dim_M + j]);
                    if (tau >= ZERO_f)
                        t = ONE_f / (tau + sqrtf(ONE_f + tau * tau));
                    else
                        t = ONE_f / (tau - sqrtf(ONE_f + tau * tau));

                    cos = ONE_f / sqrtf(ONE_f + t * t);
                    sin = t * cos;

                    /* Update rows i and j of MAT */
                    for (uint32_t m = 0; m < dim_M; m++) {
                        float im;
                        float jm;

                        im = mat[i * dim_M + m];
                        jm = mat[j * dim_M + m];

                        mat[i * dim_M + m] = (cos * im) - (sin * jm);
                        mat[j * dim_M + m] = (sin * im) + (cos * jm);
                    }
                }

                snrt_partial_barrier(&barr, 8);

                if (compute) {
                    /* Update cols i and j of MAT */
                    for (uint32_t n = 0; n < dim_M; n++) {
                        float ni;
                        float nj;

                        ni = mat[n * dim_M + i];
                        nj = mat[n * dim_M + j];

                        mat[n * dim_M + i] = (cos * ni) - (sin * nj);
                        mat[n * dim_M + j] = (sin * ni) + (cos * nj);
                    }

                    /* Update cols i and j of V */
                    for (uint32_t n = 0; n < dim_M; n++) {
                        float ni;
                        float nj;

                        ni = mat_V[n * dim_M + i];
                        nj = mat_V[n * dim_M + j];

                        mat_V[n * dim_M + i] = (cos * ni) - (sin * nj);
                        mat_V[n * dim_M + j] = (sin * ni) + (cos * nj);
                    }


                    if (fabs(mat[i * dim_M + j]) > local_max[id])
                        local_max[id] = fabs(mat[i * dim_M + j]);
                }

            } /* End of pairs for this round */

            snrt_partial_barrier(&barr, 8);

            /* Reduction */
            if (id == 0) {
                max_offdiag = local_max[0];
                for (uint32_t cid = 1; cid < NUM_CORES; cid++)
                    if (local_max[cid] > max_offdiag)
                        max_offdiag = local_max[cid];
            }

            snrt_partial_barrier(&barr, 8);

            if (max_offdiag < EPSILON)
                break;

        }   /* Round */


        if (max_offdiag < EPSILON)
            break;


    }   /* Iters */

    if (iter >= MAX_ITER) {
        return -1;
    }

    uint32_t block;
    uint32_t start;
    uint32_t left;
    uint32_t end;

    block = dim_M / NUM_CORES;
    left = dim_M % NUM_CORES;
    start = id * block + (id < left ? id : left);
    end = start + block + (id < left ? 1 : 0);

    for (uint32_t i = start; i < end; i++) {
        if (mat[i * dim_M + i] > 0)
            vec_S[i] = sqrtf(mat[i * dim_M + i]);
        else
            vec_S[i] = 0;
    }

    snrt_partial_barrier(&barr, 8);

    return 0;
}
