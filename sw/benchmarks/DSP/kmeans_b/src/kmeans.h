#include <stdint.h>
#include "math.h"
#include "snrt.h"

// Forziamo l'infinito in TCDM (importante per 32-bit per evitare fetch DRAM continui)
__thread float inf = INFINITY;

// Funzione di distanza ottimizzata per 32-bit
float euclidean_distance_squared(uint32_t n_features, float* p1, float* p2) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_features; i++) {
        float diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

void kmeans_job(uint32_t n_samples, uint32_t n_features, uint32_t n_clusters, 
                uint32_t n_iter, float* samples, float* centroids) {

    uint32_t cluster_id = snrt_cluster_idx();
    uint32_t n_clusters_sys = snrt_cluster_num();
    uint32_t n_cores_per_cluster = snrt_cluster_compute_core_num();

    // Distribuzione lavoro
    uint32_t n_samples_per_cluster = n_samples / n_clusters_sys;
    uint32_t n_samples_per_core = n_samples_per_cluster / n_cores_per_cluster;

    // Gestione Memoria L1 (TCDM) - 32 bit alignment
    // Usiamo offset in float (4 byte) per semplicità aritmetica
    float* l1_ptr = (float*) snrt_l1_next();
    
    float* local_samples          = l1_ptr;
    float* local_centroids        = local_samples + (n_samples_per_cluster * n_features);
    uint32_t* membership             = (uint32_t*) (local_centroids + (n_clusters * n_features));
    uint32_t* partial_membership_cnt = (uint32_t*) (membership + n_samples_per_cluster);
    float* partial_centroids      = (float*) (partial_membership_cnt + n_clusters);

    // --- CARICAMENTO INIZIALE DMA ---
    if (snrt_is_dm_core()) {
        // Correzione Offset: samples è float*, quindi l'offset è in numero di elementi
        uint32_t offset = cluster_id * n_samples_per_cluster * n_features;
        size_t sample_size = n_samples_per_cluster * n_features * sizeof(float);
        size_t centroid_size = n_clusters * n_features * sizeof(float);
        // Carica i campioni di competenza del cluster
        snrt_dma_start_1d(local_samples, &samples[offset], sample_size);
        
        // Carica i centroidi iniziali (tutti i cluster leggono gli stessi)
        snrt_dma_start_1d(local_centroids, centroids, centroid_size);
        
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // --- LOOP ALGORITMO ---
    for (uint32_t iter = 0; iter < n_iter; iter++) {
        
        // 1. Fase di assegnamento (Computation)
        if (snrt_is_compute_core()) {
            uint32_t core_id = snrt_cluster_core_idx();
            uint32_t start_idx = core_id * n_samples_per_core;
            uint32_t end_idx = start_idx + n_samples_per_core;

            // Reset contatori parziali
            for (uint32_t i = 0; i < n_clusters; i++) {
                partial_membership_cnt[i] = 0;
                for (uint32_t j = 0; j < n_features; j++)
                    partial_centroids[i * n_features + j] = 0.0f;
            }

            for (uint32_t i = start_idx; i < end_idx; i++) {
                float min_dist = inf;
                uint32_t best_cluster = 0;

                for (uint32_t c = 0; c < n_clusters; c++) {
                    float dist = euclidean_distance_squared(n_features, 
                                 &local_samples[i * n_features], 
                                 &local_centroids[c * n_features]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }
                membership[i] = best_cluster;
                partial_membership_cnt[best_cluster]++;
                
                // Accumulo parziale locale al core
                for (uint32_t f = 0; f < n_features; f++) {
                    partial_centroids[best_cluster * n_features + f] += 
                        local_samples[i * n_features + f];
                }
            }
        }

        snrt_cluster_hw_barrier();

        // 2. Riduzione Intra-Cluster (Core 0 aggrega i dati degli altri core)
        if (snrt_is_compute_core() && snrt_cluster_core_idx() == 0) {
            for (uint32_t c_idx = 1; c_idx < n_cores_per_cluster; c_idx++) {
                uint32_t* rem_cnt = (uint32_t*) snrt_compute_core_local_ptr(partial_membership_cnt, c_idx, 0);
                float* rem_centroids = (float*) snrt_compute_core_local_ptr(partial_centroids, c_idx, 0);
                
                for (uint32_t i = 0; i < n_clusters; i++) {
                    partial_membership_cnt[i] += rem_cnt[i];
                    for (uint32_t j = 0; j < n_features; j++) {
                        partial_centroids[i * n_features + j] += rem_centroids[i * n_features + j];
                    }
                }
            }
        }

        snrt_inter_cluster_barrier();

        // 3. Riduzione Inter-Cluster (Solo Cluster 0 aggrega tutto e normalizza)
        if (cluster_id == 0 && snrt_is_compute_core() && snrt_cluster_core_idx() == 0) {
            for (uint32_t cl_idx = 1; cl_idx < n_clusters_sys; cl_idx++) {
                uint32_t* rem_cnt = (uint32_t*) snrt_remote_l1_ptr(partial_membership_cnt, 0, cl_idx);
                float* rem_centroids = (float*) snrt_remote_l1_ptr(partial_centroids, 0, cl_idx);
                
                for (uint32_t i = 0; i < n_clusters; i++) {
                    partial_membership_cnt[i] += rem_cnt[i];
                    for (uint32_t j = 0; j < n_features; j++) {
                        partial_centroids[i * n_features + j] += rem_centroids[i * n_features + j];
                    }
                }
            }

            // Normalizzazione: Calcolo media finale (nuovi centroidi)
            for (uint32_t i = 0; i < n_clusters; i++) {
                if (partial_membership_cnt[i] > 0) {
                    for (uint32_t j = 0; j < n_features; j++) {
                        partial_centroids[i * n_features + j] /= partial_membership_cnt[i];
                    }
                }
            }
        }

        snrt_inter_cluster_barrier();

        // 4. BROADCAST: Copia i nuovi centroidi da Cluster 0 a tutti i local_centroids
        // Questo risolve il problema RegWriteKnown dell'iterazione successiva
        if (snrt_is_dm_core()) {
            float* src_new_centroids = (float*) snrt_remote_l1_ptr(partial_centroids, 0, 0);
            size_t centroid_size = n_clusters * n_features * sizeof(float);
            snrt_dma_start_1d(local_centroids, src_new_centroids, centroid_size);
            snrt_dma_wait_all();
        }
        
        snrt_cluster_hw_barrier();
    }

    // --- SCRITTURA FINALE ---
    if (cluster_id == 0 && snrt_is_dm_core()) {
        size_t centroid_size = n_clusters * n_features * sizeof(float);
        snrt_dma_start_1d(centroids, local_centroids, centroid_size);
        snrt_dma_wait_all();
    }
}