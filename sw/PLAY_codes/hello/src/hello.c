// Luca Colombo 2025 Chips-IT
// Hello from each core of the cluster
// In standard cluster configuration 8 worker cores (0 to 7)
// plus one DM core, number 8.

#include "snrt.h"

int main(){
    // Get the core idx
    uint32_t core_idx = snrt_cluster_core_idx();

    // If core is dm core custom printf
    if(snrt_is_dm_core()){
        printf("Hello from core %d, the DM core!\n", core_idx);
    }
    else{
        // Worker cores printf
        printf("Hello from core %d, a worker core!\n", core_idx);
    }

}