#include <math.h>
#include <stdio.h>

#include "snrt.h"
#include "data.h"
#include "attention.h"

static float *inp_tcdm;     // (B, T, 3C)
static float *out_tcdm;     // (B, T, C)
static float *preatt_tcdm;  // (B, NH, T, T)
static float *att_tcdm;     // (B, NH, T, T)

int main(void) {
    uint32_t core_idx = snrt_cluster_core_idx();

    if (snrt_is_dm_core()) {
        float *ptr = (float *)snrt_l1_next();

        inp_tcdm    = ptr;   ptr += T * 3 * C;
        out_tcdm    = ptr;   ptr += T * C;
        preatt_tcdm = ptr;   ptr += NH * T * T;
        att_tcdm    = ptr;   /* NH * T * T floats */
        
        for (int t = 0; t < T; t++) {
            // Q va in inp[t*3*C]
            snrt_dma_start_1d(inp_tcdm + t * 3 * C,         Q + t * C, C * sizeof(float));
            // K va in inp[t*3*C + C]
            snrt_dma_start_1d(inp_tcdm + t * 3 * C +     C, K + t * C, C * sizeof(float));
            // V va in inp[t*3*C + 2*C]
            snrt_dma_start_1d(inp_tcdm + t * 3 * C + 2 * C, V + t * C, C * sizeof(float));
        }
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    if (snrt_is_compute_core()) {
        for(uint32_t i=0; i<4; i++){
            attention( out_tcdm, preatt_tcdm, att_tcdm,
                inp_tcdm, (uint32_t)B, (uint32_t)T, (uint32_t)C, (uint32_t)NH);
        }
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;
    if (CHECK_RESULTS && core_idx == 0) {
        const float eps = 1e-3f; // High as we use fast_exp approximation
        for (uint32_t i = 0; i < (uint32_t)(T * C); i++) {
            if (fabsf(out_tcdm[i] - O_golden[i]) > eps)
                err++;
        }
        printf("Errors: %u\n", err);
    }

    return (int)err;
}