#include "math.h"
#include "snrt.h"

__thread float inf = INFINITY;


float euclidean_distance_squared(uint32_t n_features, float* p1, float* p2) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_features; i++) {
        float diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

void kmeans_iteration(uint32_t n_samples_per_core,
                                    uint32_t n_clusters, uint32_t n_features,
                                    float* samples, uint32_t* membership,
                                    uint32_t* partial_membership_cnt,
                                    float* initial_centroids,
                                    float* partial_centroids) {
    // Distribute work
    uint32_t start_sample_idx;
    uint32_t end_sample_idx;
   
    start_sample_idx = snrt_cluster_core_idx() * n_samples_per_core;
    end_sample_idx = start_sample_idx + n_samples_per_core;

    // Assignment step
    for (uint32_t centroid_idx = 0; centroid_idx < n_clusters;
            centroid_idx++) {
        partial_membership_cnt[centroid_idx] = 0;
    }
    snrt_fpu_fence();
    for (uint32_t sample_idx = start_sample_idx;
            sample_idx < end_sample_idx; sample_idx++) {
        float min_dist = inf;
        membership[sample_idx] = 0;

        for (uint32_t centroid_idx = 0; centroid_idx < n_clusters;
                centroid_idx++) {
            float dist = euclidean_distance_squared(
                n_features, &samples[sample_idx * n_features],
                &initial_centroids[centroid_idx * n_features]);
            if (dist < min_dist) {
                min_dist = dist;
                membership[sample_idx] = centroid_idx;
            }
        }
        partial_membership_cnt[membership[sample_idx]]++;
    }
    
    asm volatile(  
        "nop \n"
        :::);

    // Update step
    for (uint32_t centroid_idx = 0; centroid_idx < n_clusters;
            centroid_idx++) {
        for (uint32_t feature_idx = 0; feature_idx < n_features;
                feature_idx++) {
            // Initialize centroids to zero
            // TODO: Can be optimized w/ DMA
            partial_centroids[centroid_idx * n_features + feature_idx] = 0;
        }
    }
    snrt_fpu_fence();
    for (uint32_t sample_idx = start_sample_idx;
            sample_idx < end_sample_idx; sample_idx++) {
        for (uint32_t feature_idx = 0; feature_idx < n_features;
                feature_idx++) {
            partial_centroids[membership[sample_idx] * n_features +
                                feature_idx] +=
                samples[sample_idx * n_features + feature_idx];
        }
    }

    asm volatile(  
        "nop \n"
        :::);

    snrt_cluster_hw_barrier();

    
    if (snrt_cluster_core_idx() == 0) {
        // Intra-cluster reduction
        for (uint32_t core_idx = 1;
                core_idx < snrt_cluster_compute_core_num(); core_idx++) {
            // Pointers to variables of the other core
            uint32_t* remote_partial_membership_cnt = 
                    partial_membership_cnt + core_idx * n_clusters;
            float* remote_partial_centroids =
                    partial_centroids + core_idx * n_clusters * n_features;
            for (uint32_t centroid_idx = 0; centroid_idx < n_clusters;
                    centroid_idx++) {
                // Accumulate membership counters
                partial_membership_cnt[centroid_idx] +=
                    remote_partial_membership_cnt[centroid_idx];
                // Accumulate centroid features
                for (uint32_t feature_idx = 0; feature_idx < n_features;
                        feature_idx++) {
                    partial_centroids[centroid_idx * n_features +
                                        feature_idx] +=
                        remote_partial_centroids[centroid_idx * n_features +
                                                    feature_idx];
                }
            }
        }
        asm volatile(  
        "nop \n"
        :::);
        snrt_inter_cluster_barrier();

        if (snrt_cluster_idx() == 0) {
            

            // Inter-cluster reduction
            for (uint32_t cluster_idx = 1; cluster_idx < snrt_cluster_num();
                    cluster_idx++) {
                // Pointers to variables of remote clusters
                uint32_t* remote_partial_membership_cnt =
                    (uint32_t*)snrt_remote_l1_ptr(partial_membership_cnt, 0,
                                                    cluster_idx);
                float* remote_partial_centroids =
                    (float*)snrt_remote_l1_ptr(partial_centroids, 0,
                                                cluster_idx);
                for (uint32_t centroid_idx = 0; centroid_idx < n_clusters;
                        centroid_idx++) {
                    // Accumulate membership counters
                    partial_membership_cnt[centroid_idx] +=
                        remote_partial_membership_cnt[centroid_idx];
                    // Accumulate centroid features
                    for (uint32_t feature_idx = 0; feature_idx < n_features;
                            feature_idx++) {
                        partial_centroids[centroid_idx * n_features +
                                            feature_idx] +=
                            remote_partial_centroids[centroid_idx *
                                                            n_features +
                                                        feature_idx];
                    }
                }
            }
            asm volatile(  
            "nop \n"
            :::);
            // Normalize
            for (uint32_t centroid_idx = 0; centroid_idx < n_clusters;
                    centroid_idx++) {
                for (uint32_t feature_idx = 0; feature_idx < n_features;
                        feature_idx++) {
                    partial_centroids[centroid_idx * n_features +
                                        feature_idx] /=
                        partial_membership_cnt[centroid_idx];
                }
            }
        }
        asm volatile(  
        "nop \n"
        :::);
    }

}