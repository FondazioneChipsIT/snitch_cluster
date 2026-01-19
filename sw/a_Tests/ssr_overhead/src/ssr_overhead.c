#include "snrt.h"

double *test_pointer;


int main() {

    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t offset = 3;
    uint32_t m =2;
    uint32_t x =7;
    if(core_idx == 0) {


        uint64_t start = snrt_mcycle();

        asm volatile ("START:");
        // 3 SSR overheads
        snrt_ssr_loop_4d(SNRT_SSR_DM0, 64, 64, 64, 64, sizeof(double), sizeof(double), sizeof(double), sizeof(double));
        /*snrt_ssr_loop_1d(SNRT_SSR_DM1, 64, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM2, 64, sizeof(double));*/

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, test_pointer + offset*x+m);
        //snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, test_pointer +  offset*m+x);
        //snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, test_pointer +  m*x+offset); 
        
        snrt_ssr_enable();


        snrt_ssr_disable();
        
        asm volatile ("END:");

        uint64_t end = snrt_mcycle();

        uint64_t cycles = end - start;

        snrt_fpu_fence();

        printf("SSR Overhead Cycles: %llu\n", cycles);
    }
    snrt_cluster_hw_barrier();
    return 0;
}



