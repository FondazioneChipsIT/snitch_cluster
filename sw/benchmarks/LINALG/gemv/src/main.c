// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Luca Colagrande <colluca@iis.ee.ethz.ch>

#include "gemv.h"

#include "data.h"
#include "snrt.h"

int main() {
    uint32_t trans = args.trans;
    uint32_t m = args.m;
    uint32_t n = args.n;
    float alpha = args.alpha;
    float *a = args.a;
    float *x = args.x;
    float *y = args.y;

    uint32_t size_a = m * n * sizeof(float);
    uint32_t size_x = n * sizeof(float);
    uint32_t size_y = m * sizeof(float);

    float *local_a =
        (float *)snrt_l1_alloc_cluster_local(size_a, sizeof(float));
    float *local_x =
        (float *)snrt_l1_alloc_cluster_local(size_x, sizeof(float));
    float *local_y =
        (float *)snrt_l1_alloc_cluster_local(size_y, sizeof(float));

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, a, size_a);
        snrt_dma_start_1d(local_x, x, size_x);
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        gemv(trans, m, n, alpha, local_a, local_x, 1, local_y);
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(y, local_y, size_y);
    }

    return 0;
}
