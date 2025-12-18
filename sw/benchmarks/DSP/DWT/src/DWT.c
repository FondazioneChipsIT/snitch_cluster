// Luca Colombo Chips-IT 2025
/* DWT filter */

// Snitch runtime library
#include "snrt.h"
#include "data.h"
#include "DWT_opt.h"

int main(){
    // Core ID and core count
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores = snrt_cluster_compute_core_num();

    // DM core allocates memory in TCDM and initializes the vectors
    if(snrt_is_dm_core()){

        // Pointers to TCDM memory, spaced by LEN 
        x = (double *)snrt_l1_next();
        y_low = x + LEN;
        y_high = y_low + LEN/2;
        h = y_high + LEN/2;
        g = h + FILTER_LEN;

        // If pointers are null -> break
        if (!x || !y_low || !y_high || !h || !g) {
            printf("Memory allocation failed!\n");
            return -1;
        } 

        // Initialize the values of vectors, can change as you like
        for(uint32_t i = 0; i<LEN; i++){
            x[i] = (double)i;
        }
        /* Haar
        h[0] = 1.0;
        h[1] = 0.0;

        g[0] = 0.0;
        g[1] = 1.0;*/

        /* DB4

        h[0] =  0.4829629131445341;
        h[1] =  0.8365163037378079;
        h[2] =  0.2241438680420134;
        h[3] = -0.1294095225512604;

        g[0] = -0.1294095225512604;
        g[1] = -0.2241438680420134;
        g[2] =  0.8365163037378079;
        g[3] = -0.4829629131445341;*/

        // Daubechies-16 low-pass (analysis)
        size_t size = FILTER_LEN * sizeof(double);
        snrt_dma_start_1d(h, h_L2, size);
        snrt_dma_wait_all();
        
        // High pass
        for (int k = 0; k < 32; k++) {
            g[k] = ((k & 1) ? -1.0 : 1.0) * h[31 - k];
        }


    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Only the compute cores do something
    if(snrt_is_compute_core()){

        // Compute the chunk of the vector per core, and the offset that is used to space them
        uint32_t dwt_len = LEN / 2;
        uint32_t chunk_per_core = dwt_len / ncores;
        uint32_t offset = core_idx * chunk_per_core;

        // Call the kernel
        dwt_opt(chunk_per_core, offset, x, y_low, y_high, h, g);

    }

    snrt_cluster_hw_barrier(); // Barrier syncronization

    // Check correctnes of results
    if(core_idx==0){

        double energy_x = 0, energy_y = 0;
        // Db4 check
        /* Skip the first truncated
        for(uint32_t i=FILTER_LEN;i<LEN;i++)
            energy_x += x[i]*x[i];

        for(uint32_t i=FILTER_LEN;i<LEN/2;i++)
            energy_y += y_low[i]*y_low[i] + y_high[i]*y_high[i];

        double eps = 1e-9;
        if (fabs(energy_x - energy_y) > eps * energy_x) {
            printf("Energy mismatch!\n");
            return 22;
        }*/

        // Haar check
        /*for(uint32_t i=0;i<20;i++){

            printf("Y_low %d, %f\n",i,y_low[i]);

            printf("Y_high %d, %f\n",i,y_high[i]);
        }*/
    }


    return 0;
}

/* NEVER USE FLOAT TYPE! BREAKS EVERYTHING! */