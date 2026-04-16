// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "data.h"
#include "kmeans_new.h"

uint32_t CHECK_RESULTS = 1; // Set to 0 to skip verification (for performance measurement)

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t n_cores  = snrt_cluster_compute_core_num();

    // Distribuzione campioni con remainder corretto
    uint32_t chunk     = n_samples / n_cores;
    uint32_t remainder = n_samples % n_cores;
    uint32_t start_idx = core_idx * chunk + (core_idx < remainder ? core_idx : remainder);
    uint32_t end_idx   = start_idx + chunk + (core_idx < remainder ? 1 : 0);

    // ── Layout TCDM (un'unica allocazione contigua) ───────────────────────────
    //  local_samples       [n_samples * n_features]           float
    //  local_centroids     [n_clusters * n_features]          float  ← aggiornato in-place
    //  partial_centroids   [n_cores * n_clusters * n_features] float
    //  membership          [n_samples]                        uint32_t
    //  partial_cnt         [n_cores * n_clusters]             uint32_t

    float*    local_samples   = (float*)snrt_l1_next();
    float*    local_centroids = local_samples   + n_samples  * n_features;
    float*    partial_cents   = local_centroids + n_clusters * n_features;
    uint32_t* membership      = (uint32_t*)(partial_cents + n_cores * n_clusters * n_features);
    uint32_t* partial_cnt     = membership + n_samples;

    // ── DMA: carica campioni e centroidi iniziali ─────────────────────────────
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_samples,   (float*)samples,
                          n_samples  * n_features * sizeof(float));
        snrt_dma_start_1d(local_centroids, (float*)centroids,
                          n_clusters * n_features * sizeof(float)); // ← era *sizeof(float) due volte
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    snrt_mcycle();

    if (snrt_is_compute_core()) {
        // Ogni core punta alla propria fetta degli array parziali
        float*    my_partial = partial_cents + core_idx * n_clusters * n_features;
        uint32_t* my_cnt     = partial_cnt   + core_idx * n_clusters;

        for (uint32_t iter = 0; iter < n_iter; iter++) {
            kmeans_iteration(
                start_idx, end_idx,
                n_clusters, n_features,
                local_samples, membership,
                my_cnt,     local_centroids,
                my_partial, partial_cnt, partial_cents
            );
            // local_centroids viene aggiornato in-place da core 0 dentro kmeans_iteration
        }
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;
    float eps = 0.1f;

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        for(uint32_t i = 0; i < n_clusters * n_features; i++){
            if(fabsf(local_centroids[i] - golden_centroids[i]) > eps){ // Using a tolerance of 0.1 for classification output
                err ++;
            }
        }
    }

    return err;
}