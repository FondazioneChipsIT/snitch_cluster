#include "snrt.h"
#include "data_def.h"
#include "out_ref.h"
#include "kmeans_new.h"

uint32_t CHECK_RESULTS = 1;

uint32_t max_iter = 1000;

int main() {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t n_cores  = snrt_cluster_compute_core_num();

    uint32_t chunk     = n_samples / n_cores;
    uint32_t remainder = n_samples % n_cores;
    uint32_t start_idx = core_idx * chunk + (core_idx < remainder ? core_idx : remainder);
    uint32_t end_idx   = start_idx + chunk + (core_idx < remainder ? 1 : 0);

    // ── Layout TCDM ──────────────────────────────────────────────────────────
    // Barrier in TCDM (see the note in kmeans_new.h): here the layout is
    // computed by every core, so the barrier is simply the first object of it
    // and only the DM core initialises it below.
    snrt_barrier_t* l1_barr   = (snrt_barrier_t*)snrt_l1_next();
    float*    local_samples   = (float*)(l1_barr + 1);
    float*    local_delta = local_samples + n_samples * n_features;
    float*    local_centroids = local_delta   + n_cores;
    float*    partial_cents   = local_centroids + n_clusters * n_features;
    uint32_t* membership      = (uint32_t*)(partial_cents + n_cores * n_clusters * n_features);
    uint32_t* partial_cnt     = membership + n_samples;;
    float*    golden_L1 = (float*)partial_cnt + n_cores * n_clusters;

    float zero[8] = {0.0f};
    // Init membership to 0
    for (int i = 0; i < n_samples; i++) {
        membership[i] = 0;
    }
    // ── DMA ──────────────────────────────────────────────────────────────────
    if (snrt_is_dm_core()) {
        barr = l1_barr;
        barr->cnt = 0;
        barr->iteration = 0;

        snrt_dma_start_1d(local_samples,   (float*)samples,
                          n_samples  * n_features * sizeof(float));
        snrt_dma_start_1d(local_centroids, (float*)centroids,
                          n_clusters * n_features * sizeof(float));
        snrt_dma_start_1d(local_delta, (float*)zero,
                          n_cores * sizeof(float));
        snrt_dma_start_1d(golden_L1, (float*)golden_centroids,
                          n_clusters * n_features * sizeof(float));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    if (snrt_is_compute_core()) {
        float*    my_partial = partial_cents + core_idx * n_clusters * n_features;
        uint32_t* my_cnt     = partial_cnt   + core_idx * n_clusters;

            kmeans_iteration(
                start_idx, end_idx,
                n_clusters, n_features,
                local_samples, membership,
                my_cnt,        local_centroids,
                my_partial,    partial_cnt, partial_cents, max_iter, n_samples,
                local_delta
            );
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    uint32_t err = 0;
    float    eps = threshold; 
    
    if (CHECK_RESULTS == 1 && core_idx == 0) {
        uint32_t used[8] = {0}; 
        for (uint32_t k = 0; k < n_clusters; k++) {
            float    best_dist = __builtin_inff();
            uint32_t best_g    = 0;

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

            for (uint32_t f = 0; f < n_features; f++) {
                if (fabsf(local_centroids[k * n_features + f]
                        - golden_centroids[best_g * n_features + f]) > eps)
                    err++;
            }
        }
    }
    
    return err;
}