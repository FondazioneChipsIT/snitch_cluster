// Luca Colombo Chips-IT 2025
// NAIVE multicore LU solve version

/*

# Lu solve 64x64 performance
# Mean cycles: 65103
# Mean FLOP/cycle: 0.124848
# Total FLOP/cycle: 0.998781

# Lu solve 32x32 performance
# Mean cycles: 31542
# Mean FLOP/cycle: 0.063913
# Total FLOP/cycle: 0.511306


*/



void lu_solve_naive(uint64_t *start_cycle, uint64_t *end_cycle,
                    float *mat, uint32_t *perm, float *y, float *vec, float *result,
                    float *local_sum) {


    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores   = snrt_cluster_compute_core_num();
    uint32_t perm_idx;

    *start_cycle = snrt_mcycle();

    // -------------------------
    // FORWARD SUBSTITUTION (L * y = P * vec)
    // -------------------------
    for (uint32_t m = 0; m < elems; m++) {
        if(snrt_is_compute_core()){
            // reset buffer
            local_sum[core_idx] = 0.0f;
            

                // numero di elementi da sommare: k in [0, m)
            int count = m; // può essere 0
            int block = count / (int)ncores;
            int left  = count % (int)ncores;

            // start/end per questo core (distribuzione dei resti sui primi 'left' core)
            int start = core_idx * block + (core_idx < (uint32_t)left ? core_idx : left);
            int end   = start + block + (core_idx < (uint32_t)left ? 1 : 0);

            for (uint32_t k = start; k < end; k++)
                local_sum[core_idx] += mat[m * elems + k] * y[k];
        }

        // barrier: attendi che tutti i core finiscano
        snrt_cluster_hw_barrier();

        // core 0 somma tutte le parti e aggiorna y[m]
        if (core_idx == 0) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < ncores; i++)
                sum += local_sum[i];

            perm_idx = perm[m];
            y[m] = vec[perm_idx] - sum;
        }
        
        // barrier: tutti i core attendono che y[m] sia calcolato
        snrt_cluster_hw_barrier();
    }

    snrt_cluster_hw_barrier();


    // BACKWARD SUBSTITUTION (U * result = y)
    
    for (int m = (int) elems - 1; m >=0; m--) {
        if(snrt_is_compute_core()){
            local_sum[core_idx] = 0.0;
            

             // la finestra su cui sommare è k in [m+1, elems)
            int count = (int)elems - (m + 1); // numero elementi nella finestra, può essere 0
            int block = count / (int)ncores;
            int left  = count % (int)ncores;

            int start_rel = core_idx * block + (core_idx < (uint32_t)left ? core_idx : left);
            int end_rel   = start_rel + block + (core_idx < (uint32_t)left ? 1 : 0);

            int start = (m + 1) + start_rel;
            int end   = (m + 1) + end_rel;

            for (int k = start; k < end; k++)
                local_sum[core_idx] += mat[m * elems + k] * result[k];
        }

        snrt_cluster_hw_barrier();

        // core 0 somma e calcola result[m]
        if (core_idx == 0) {
            float sum = 0.0;
            for (uint32_t i = 0; i < ncores; i++)
                sum += local_sum[i];
            result[m] = (y[m] - sum) / mat[m * elems + m];
        }
     
        snrt_cluster_hw_barrier();
    }

    *end_cycle = snrt_mcycle();

    return;
}
