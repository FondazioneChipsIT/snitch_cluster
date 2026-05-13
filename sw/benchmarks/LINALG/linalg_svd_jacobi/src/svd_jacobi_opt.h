#include <math.h>
#include "snrt.h"

uint32_t MAX_ITER = 1000;
float EPSILON = 1e-4f;

snrt_barrier_t   barr;
volatile float   max_offdiag;
volatile float max_scale_global;
volatile float local_scale[8];


void svd_jacobi_opt(float *mat, float *mat_V, float *vec_S,
                    uint32_t dim_M, uint32_t dim_N) {
    const float zero = 0.0f;
    const float one  = 1.0f;
    const float two  = 2.0f;

    uint32_t core_idx  = snrt_cluster_core_idx();
    uint32_t NUM_CORES = snrt_cluster_compute_core_num();

    uint32_t pairs_per_round = dim_N / 2;
    uint32_t num_rounds      = dim_N - 1;
    uint32_t iter;

    bool done = false;

    for (iter = 0; ((iter < MAX_ITER) && (!done)); iter++) {

        for (uint32_t round = 0; ((round < num_rounds) && (!done)); round++) {

            /* Reset accumulatore locale (obbligatorio ad ogni round!) */
            local_max[core_idx] = zero;
            local_scale[core_idx] = zero;

            /* ── Distribuzione bilanciata delle coppie ai core ─────────── */
            uint32_t chunk  = pairs_per_round / NUM_CORES;
            uint32_t rem    = pairs_per_round % NUM_CORES;
            uint32_t offset = core_idx * chunk + (core_idx < rem ? core_idx : rem);
            uint32_t count  = chunk    + (core_idx < rem ? 1u   : 0u);


            for (uint32_t k = 0; k < count; k++) {
                uint32_t pair = offset + k;

                uint32_t i = (round + pair) % (dim_N - 1);
                uint32_t j = (pair == 0)
                           ? (dim_N - 1)
                           : ((dim_N - 1 - pair + round) % (dim_N - 1));

                if (i >= dim_N || j >= dim_N || i == j) continue;

                float alpha = zero, beta = zero, gamma = zero;
                for (uint32_t m = 0; m < dim_M; m++) {
                    float ai = mat[m * dim_N + i];
                    float aj = mat[m * dim_N + j];
                    alpha += ai * ai;
                    beta  += aj * aj;
                    gamma += ai * aj;
                }

                float abs_gamma = fabsf(gamma);

                if (abs_gamma > local_max[core_idx])
                    local_max[core_idx] = abs_gamma;

                float scale = sqrtf(alpha * beta + EPSILON);
                if (scale > local_scale[core_idx])
                    local_scale[core_idx] = scale;

                /* ── Criterio di convergenza relativo ────────────────────
                 *   +EPSILON al denominatore evita divisione per zero.
                 * ─────────────────────────────────────────────────────── */
                if (abs_gamma <= EPSILON * scale)
                    continue;


                float zeta = (beta - alpha) / (two * gamma);
                float t;

                if (zeta >= zero)
                    t =  one / ( zeta + sqrtf(one + zeta * zeta));
                else
                    t = -one / (-zeta + sqrtf(one + zeta * zeta));

                float c = one / sqrtf(one + t * t);
                float s = t * c;

                /* Is it possibile to use SSRs?
                I dont think so, as we would need at leas 2 explicit memory ops, 
                as we have 3 ssrs lane and we need to read and write mat
                We could setup, read with ssrs, compute, and setup ssrs for write, but
                the lost cycles for setup and the fact that we have to read and write 2 columns of mat, makes me think that it is not worth it */
                for (uint32_t m = 0; m < dim_M; m++) {
                    float ai = mat[m * dim_N + i];
                    float aj = mat[m * dim_N + j];
                    mat[m * dim_N + i] = c * ai - s * aj;
                    mat[m * dim_N + j] = s * ai + c * aj;
                }

                /* ── Aggiorna colonne i, j di mat_V [dim_N × dim_N] ───── */
                for (uint32_t n = 0; n < dim_N; n++) {
                    float vi = mat_V[n * dim_N + i];
                    float vj = mat_V[n * dim_N + j];
                    mat_V[n * dim_N + i] = c * vi - s * vj;
                    mat_V[n * dim_N + j] = s * vi + c * vj;
                }

            } /* fine ciclo coppie */

            /* ── Barrier + riduzione del massimo off-diagonale ──────────── */
            snrt_partial_barrier(&barr, 8);

            if (core_idx == 0) {
                float mx = local_max[0];
                float sc = local_scale[0];
                for (uint32_t cid = 1; cid < 8; cid++){
                    if (local_max[cid] > mx) mx = local_max[cid];
                    if (local_scale[cid] > sc) sc = local_scale[cid];
                }
                max_scale_global = sc;
                max_offdiag = mx;
            }

            snrt_partial_barrier(&barr, 8);

            if (max_offdiag <= EPSILON * max_scale_global)
                done = true;

        } /* fine round */
    } 

    
    
    uint32_t chunk = dim_N / NUM_CORES;
    uint32_t rem   = dim_N % NUM_CORES;
    uint32_t start = core_idx * chunk + (core_idx < rem ? core_idx : rem);
    uint32_t end   = start + chunk    + (core_idx < rem ? 1u      : 0u);

    for (uint32_t i = start; i < end; i++) {
        float norm_sq = zero;
        for (uint32_t m = 0; m < dim_M; m++) {
            float ai = mat[m * dim_N + i];
            norm_sq += ai * ai;
        }
        vec_S[i] = (norm_sq > zero) ? sqrtf(norm_sq) : zero;
    }

    if(core_idx == 0){
        printf("Num of iter: %u\n", iter);
    }
    snrt_partial_barrier(&barr, 8);

    return;
}