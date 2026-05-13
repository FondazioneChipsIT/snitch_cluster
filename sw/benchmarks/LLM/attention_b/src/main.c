#include <math.h>
#include <stdio.h>

#include "snrt.h"
#include "data.h"
#include "attention.h"

_Static_assert(L == S, "attention_forward richiede T == S (L == S)");

static float *inp_tcdm;     // (1, L, 3D)
static float *out_tcdm;     // (1, L, D)
static float *preatt_tcdm;  // (1, 1, L, L)
static float *att_tcdm;     // (1, 1, L, L)

int main(void) {
    uint32_t core_idx = snrt_cluster_core_idx();

    // ── 1. DMA in + packing Q/K/V → inp interleaved ─────────────────────────
    if (snrt_is_dm_core()) {
        float *ptr = (float *)snrt_l1_next();

        float *Q_tmp = ptr;  ptr += L * D;
        float *K_tmp = ptr;  ptr += S * D;
        float *V_tmp = ptr;  ptr += S * D;

        inp_tcdm    = ptr;   ptr += L * 3 * D;
        out_tcdm    = ptr;   ptr += L * D;
        preatt_tcdm = ptr;   ptr += L * L;
        att_tcdm    = ptr;   /* L * L floats */

        snrt_dma_start_1d(Q_tmp, Q, L * D * sizeof(float));
        snrt_dma_start_1d(K_tmp, K, S * D * sizeof(float));
        snrt_dma_start_1d(V_tmp, V, S * D * sizeof(float));
        snrt_dma_wait_all();

        for (int t = 0; t < L; t++) {
            snrt_dma_start_1d(inp_tcdm + t * 3 * D,         Q_tmp + t * D, D * sizeof(float));
            snrt_dma_start_1d(inp_tcdm + t * 3 * D +     D, K_tmp + t * D, D * sizeof(float));
            snrt_dma_start_1d(inp_tcdm + t * 3 * D + 2 * D, V_tmp + t * D, D * sizeof(float));
        }
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();


    if (snrt_is_compute_core()) {
        attention_forward(
            out_tcdm, preatt_tcdm, att_tcdm,
            inp_tcdm,
            /*B=*/1, /*T=*/L, /*C=*/D, /*NH=*/1);
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // ── 3. DMA out ───────────────────────────────────────────────────────────
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(O, out_tcdm, L * D * sizeof(float));
        snrt_dma_wait_all();
    }

    snrt_global_barrier();

    uint32_t err = 0;
    if (CHECK_RESULTS && core_idx == 0) {
        const float eps = 1e-4f;
        for (uint32_t i = 0; i < (uint32_t)(L * D); i++) {
            if (fabsf(O[i] - O_golden[i]) > eps)
                err++;
        }
        printf("Errors: %u\n", err);
    }

    return (int)err;
}