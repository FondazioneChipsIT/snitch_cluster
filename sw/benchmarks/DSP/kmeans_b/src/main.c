// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#include <stdint.h>

#include "data.h"
#include "kmeans_new.h"

int main (){

    float* samples_addr = (float*)(samples);
    float* centroids_addr = (float*)(centroids);

    // Distribute work
    uint32_t n_samples_per_cluster = n_samples / snrt_cluster_num();

    uint32_t remaining_samples = n_samples % snrt_cluster_num();

    // Split the remaining samples among the first few clusters
    uint32_t n_samples_per_core = snrt_cluster_idx() < remaining_samples ? n_samples_per_cluster/ snrt_cluster_compute_core_num() + 1 : n_samples_per_cluster/ snrt_cluster_compute_core_num();

    // Dynamically allocate space in TCDM
    float* local_samples = (float*)snrt_l1_alloc_cluster_local(
        n_samples_per_cluster * n_features * sizeof(float), sizeof(float));
    float* local_centroids = (float*)snrt_l1_alloc_cluster_local(
        n_clusters * n_features * sizeof(float), sizeof(float));
    uint32_t* membership = (uint32_t*)snrt_l1_alloc_cluster_local(
        n_samples_per_cluster * sizeof(uint32_t), sizeof(uint32_t));
    // Allocate in the cores
    uint32_t* partial_membership_cnt = (uint32_t*)snrt_l1_next(); 
    // First core's partial centroids will store final centroids
    float* partial_centroids = (float*)(partial_membership_cnt + n_clusters * sizeof(uint32_t));
    float* final_centroids = partial_centroids + n_clusters * n_features;
    final_centroids =
        (float*)snrt_remote_l1_ptr(final_centroids, snrt_cluster_idx(), 0);

    // Transfer samples and initial centroids with DMA
    size_t size;
    size_t offset;
    if (snrt_is_dm_core()) {
        size = n_samples_per_cluster * n_features;
        offset = snrt_cluster_idx() * size;
        snrt_dma_start_1d(local_samples, samples_addr + offset,
                          size * sizeof(float));
        size = n_clusters * n_features * sizeof(float);
        snrt_dma_start_1d(local_centroids, centroids_addr, size * sizeof(float));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    if(snrt_is_compute_core()) {
        // Iterations of Lloyd's K-means algorithm
        for (uint32_t iter_idx = 0; iter_idx < n_iter; iter_idx++) {
            kmeans_iteration(n_samples_per_core, n_clusters, n_features,
                            local_samples, membership, partial_membership_cnt,
                            local_centroids, partial_centroids);
            snrt_global_barrier();
            local_centroids = final_centroids;
        }
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    // Transfer final centroids with DMA
    /*if (snrt_is_dm_core()) {
        snrt_dma_start_1d((void*)centroids, (void*)final_centroids, size);
        snrt_dma_wait_all();
    }*/

    return 0;
}