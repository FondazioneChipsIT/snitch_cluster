// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#include "gemv.h"

#include "data.h"
#include "snrt.h"

// Cycle and performance metrics
uint64_t start_cycle[16], end_cycle[16];// Increase the num if ncores>16
uint64_t total_cycles[16]; 
double flop_cycle[16];


int main() {
    uint32_t trans = args.trans;
    uint32_t m = args.m;
    uint32_t n = args.n;
    double alpha = args.alpha;
    double *a = args.a;
    double *x = args.x;
    double *y = args.y;


    uint32_t size_a = m * n * sizeof(double);
    uint32_t size_x = n * sizeof(double);
    uint32_t size_y = m * sizeof(double);

    double *local_a =
        (double *)snrt_l1_alloc_cluster_local(size_a, sizeof(double));
    double *local_x =
        (double *)snrt_l1_alloc_cluster_local(size_x, sizeof(double));
    double *local_y =
        (double *)snrt_l1_alloc_cluster_local(size_y, sizeof(double));

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, a, size_a);
        snrt_dma_start_1d(local_x, x, size_x);
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        uint32_t core_idx = snrt_cluster_core_idx();

        gemv(trans, m, n, alpha, local_a, local_x, 1, local_y, &start_cycle[core_idx], &end_cycle[core_idx]);

        uint32_t ncores = snrt_cluster_compute_core_num();
        uint32_t chunk_per_core = m*n / ncores;

        if(core_idx== ncores-1 && m % ncores !=0){
            total_cycles[core_idx] = end_cycle[core_idx] - start_cycle[core_idx];
            flop_cycle[core_idx] = (((double)chunk_per_core + m%ncores)* 3.0) / (double)total_cycles[core_idx];
        }
        else{
            total_cycles[core_idx] = end_cycle[core_idx] - start_cycle[core_idx];
            flop_cycle[core_idx] = ((double)chunk_per_core * 3.0) / (double)total_cycles[core_idx];
        }

    }

    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(y, local_y, size_y);
        
    }

    if(snrt_cluster_core_idx()==0){

        uint32_t ncores = snrt_cluster_compute_core_num();
        // Mean performance values
        uint64_t mean_cycles=0;
        double mean_flop_cycle = 0.0;

        for(uint32_t cid = 0; cid < ncores; cid ++){
            mean_cycles += total_cycles[cid];
            mean_flop_cycle += flop_cycle[cid];
        }
        mean_cycles /= ncores;
        mean_flop_cycle /= ncores;

        printf("Linear algebra GEMV performance\n");
        printf("Mean cycles: %llu\n", (unsigned long long)mean_cycles);
        printf("Mean FLOP/cycle: %f\n", mean_flop_cycle);
    }
    
    return 0;
}
