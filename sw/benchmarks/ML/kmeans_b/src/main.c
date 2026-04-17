// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include <math.h>
#include "data.h"
#include "kmeans_new.h"

uint32_t CHECK_RESULTS = 1;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t n_cores  = snrt_cluster_compute_core_num();

    uint32_t chunk     = n_samples / n_cores;
    uint32_t remainder = n_samples % n_cores;
    uint32_t start_idx = core_idx * chunk + (core_idx < remainder ? core_idx : remainder);
    uint32_t end_idx   = start_idx + chunk + (core_idx < remainder ? 1 : 0);

    // ── Layout TCDM ──────────────────────────────────────────────────────────
    float*    local_samples   = (float*)snrt_l1_next();
    float*    local_centroids = local_samples   + n_samples  * n_features;
    float*    partial_cents   = local_centroids + n_clusters * n_features;
    uint32_t* membership      = (uint32_t*)(partial_cents + n_cores * n_clusters * n_features);
    uint32_t* partial_cnt     = membership + n_samples;

    // ── DMA ──────────────────────────────────────────────────────────────────
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_samples,   (float*)samples,
                          n_samples  * n_features * sizeof(float));
        snrt_dma_start_1d(local_centroids, (float*)centroids,
                          n_clusters * n_features * sizeof(float));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    snrt_mcycle();

    if (snrt_is_compute_core()) {
        float*    my_partial = partial_cents + core_idx * n_clusters * n_features;
        uint32_t* my_cnt     = partial_cnt   + core_idx * n_clusters;

        for (uint32_t iter = 0; iter < n_iter; iter++) {
            kmeans_iteration(
                start_idx, end_idx,
                n_clusters, n_features,
                local_samples, membership,
                my_cnt,        local_centroids,
                my_partial,    partial_cnt, partial_cents
            );
        }
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;
    float    eps = 1.0f; // float32 vs float64 accumula differenze su N iterazioni

    if (CHECK_RESULTS == 1 && core_idx == 0) {
        // ── Matching nearest-centroid ─────────────────────────────────────────
        // Per ogni centroide calcolato, trova il golden più vicino (L2).
        // Robusto alla label permutation indipendentemente dall'ordinamento.
        uint32_t used[8] = {0}; // n_clusters <= 8, VLA non serve

        for (uint32_t k = 0; k < n_clusters; k++) {
            float    best_dist = __builtin_inff();
            uint32_t best_g    = 0;

            // Trova il golden non ancora usato più vicino al centroide k
            for (uint32_t g = 0; g < n_clusters; g++) {
                if (used[g]) continue;
                float dist = 0.0f;
                for (uint32_t f = 0; f < n_features; f++) {
                    float d = local_centroids[k * n_features + f]
                            - golden_centroids[g * n_features + f];
                    dist += d * d;
                }
                if (dist < best_dist) { best_dist = dist; best_g = g; }
            }
            used[best_g] = 1;

            // Controlla ogni feature del match trovato
            for (uint32_t f = 0; f < n_features; f++) {
                if (fabsf(local_centroids[k * n_features + f]
                        - golden_centroids[best_g * n_features + f]) > eps)
                    err++;
            }
        }
    }

    return err;
}