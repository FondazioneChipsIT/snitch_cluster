// Luca Colombo Chips-IT 2025
/* Find the minimum value in a src vector  */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "vect_min_opt.h"
int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    // Compute the chunk of the vector per core, and the offset that is used to space them
    uint32_t chunk_per_core = LEN/ncores;
    if(chunk_per_core == 0){
        printf("Chunk for each core is 0!\n");
        return -2;
    }

    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        src = (double *)snrt_l1_next();
        min = src + LEN;
        min_core = min + 1;

        // If pointers are null -> break
        if (!src || !min || !min_core) {
            printf("Memory allocation failed!\n");
            return -1;
        } 

        // Initialize the values of vectors, can change as you like (yes not ideal for minimum)
        for(uint32_t i = 0; i<LEN; i++){
            src[i] = 13.212312314554*(double)LEN - (double) i;
        }
    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Offset to index the correct chunk of data per core
        uint32_t offset = core_idx*chunk_per_core;
        // Call the kernel
        vect_min_opt(core_idx, chunk_per_core, offset, 
                    &start_cycle[core_idx], src, min_core);
    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Core 0 checks which is the minimum between the cores results
    if(core_idx==0){
        // Loads first value of minimum values in ft3
        asm volatile(
        "fld ft3, 0(%[base])\n"    // ft3 = src[offset]
        :
        : [base] "r"(min_core)
        : "ft3", "memory");
        
        // Skip the first element if min core as it is already in ft3
        snrt_ssr_loop_1d(SNRT_SSR_DM0, ncores-1, sizeof(double));

        // Read from ft0 
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, min_core+1); //ft0<-mincore

        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_cores], 1, 0, 0 \n"
            "fmin.d ft3, ft3, ft0\n"
            :
            : [ n_cores ] "r"(ncores - 2)
            : "ft0", "ft3", "memory");
        
        snrt_ssr_disable();

        // Save final minimum result in shared memory
        asm volatile("fsd ft3, (%[res])" :: [res] "r"(min) : "memory");

        end_cycle = snrt_mcycle();
    }
    
    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Performance calculations for each core
    if(snrt_is_compute_core()){
        total_cycles[core_idx] = end_cycle-start_cycle[core_idx];
        // If we consider core 0 it performs ncores-1 flop more, not relevant for big vectors
        if(core_idx==0){
            flop_cycle[core_idx] = (double) (chunk_per_core + ncores-1)/ (double) total_cycles[core_idx];
        }else{
            flop_cycle[core_idx] = (double) chunk_per_core/ (double) total_cycles[core_idx];
        }
    }

    snrt_cluster_hw_barrier(); // Barrier syncronization
            
    if(core_idx==0){

        // Mean performance values
        uint64_t mean_cycles=0;
        double mean_flop_cycle = 0.0;

        for(uint32_t cid = 0; cid < ncores; cid ++){
            mean_cycles += total_cycles[cid];
            mean_flop_cycle += flop_cycle[cid];
        }
        mean_cycles /= ncores;
        mean_flop_cycle /= ncores;

        printf("Vector offset %d performance\n",LEN);
        printf("Mean cycles: %llu\n", (unsigned long long)mean_cycles);
        printf("Mean FLOP/cycle: %f\n", mean_flop_cycle);
        printf("Minimum value: %f\n", *min);

    }
    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */