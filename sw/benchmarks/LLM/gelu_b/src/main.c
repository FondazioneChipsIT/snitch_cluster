// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "dnn.h"

#include "data.h"

uint32_t CHECK_RESULTS = 1;

int main() {
    gelu_layer(layer);

    uint32_t err = 0;
    float eps = 1e-5f;

    // No golden vector is generated for this benchmark, so the reference is
    // recomputed here with the same scalar formula the kernel implements.
    if (CHECK_RESULTS == 1 && snrt_cluster_core_idx() == 0) {
        for(uint32_t i = 0; i < layer.size; i++){
            float ref = sigmoid_gelu_fp32(layer.ifmap[i], -0.2888f, -1.769f);
            if(fabsf(layer.ofmap[i] - ref) > eps){
                err ++;
            }
        }
        printf("Errors: %u\n", err);
    }

    return err;
}
