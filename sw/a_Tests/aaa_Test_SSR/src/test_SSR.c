// Luca Colombo Chips-IT 2025
// A single core vector add to test SSRs and TCDM allocation

#include "snrt.h"

// Vector length
uint32_t LEN = 16;
double *a,*b,*sum;

int main(){
    // Core ID and cycle counters
    uint32_t core_idx = snrt_cluster_core_idx();
    uint64_t start_cycle, end_cycle;

    if(snrt_is_dm_core()){
        printf("Hi DM core!\n");

        // Pointers to TCDM memory
        a = (double *)snrt_l1_next();
        b = a + LEN;
        sum = b + LEN;

        if (!a || !b || !sum) {
            printf("Memory allocation failed!\n");
            return -1;
        } 

        // Initialize the values of vectors
        for(uint32_t i = 0; i<LEN; i++){
            a[i] = (double)i+1;
            b[i] = (double)i+1;
            sum[i] = 0.0;
        }

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only core 0 does something
    if(core_idx==0){

        printf("Hi from core %d\n",core_idx);

        // Setup the 1d loop with ssr (tell which streams to use, the size and the size of
        // the elements)
        snrt_ssr_loop_1d(SNRT_SSR_DM0, LEN, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, LEN, sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM2, LEN, sizeof(double));

        // Read from ft0 and ft1 that will be wired to a and b 
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, a); //ft0
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, b); //ft1
        // Write to ft2 (sum)
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, sum);

        start_cycle = snrt_mcycle();
        // Enable the SSRs
        snrt_ssr_enable();
        
        // Assembly code to add the ft0 and ft1 to ft2
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"
            "fadd.d ft2, ft0, ft1\n"
            :
            : [ n_frep ] "r"(LEN - 1)
            : "ft0", "ft1", "ft2", "memory");

        // Disable SSRs
        snrt_ssr_disable();
        
        snrt_fpu_fence();

        end_cycle = snrt_mcycle();
        
        uint64_t total_cycles = end_cycle-start_cycle;
        double flop_cycle = (double) LEN/ (double) total_cycles;

        // Print performance metrics
        printf("Cycles: %llu \n",(unsigned long long)total_cycles);
        printf("Flop/cycle: %f \n", flop_cycle);
        

        // Print results
        if(LEN<17){
            for(uint32_t i=0; i<LEN; i++){
                printf("Sum(%d): %f\n",i, sum[i]);
            }
        }
        
    }
    return 0;
}