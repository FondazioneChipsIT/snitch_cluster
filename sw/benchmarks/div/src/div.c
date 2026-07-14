// Luca Colombo Chips-IT 2026

#include "snrt.h"
int main() {
    uint32_t core_idx = snrt_cluster_core_idx();

    if( core_idx == 0){
        float a = 1.0;
        asm volatile(
        "flw ft0, 0(%[a])\n"
        "fdiv.s ft1, ft0, ft0\n"
        :: [a] "r"(&a)
        : "ft0", "ft1");
    }
    return 0;
}
