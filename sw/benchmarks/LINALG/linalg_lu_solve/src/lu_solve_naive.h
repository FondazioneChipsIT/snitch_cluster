// Luca Colombo Chips-IT 2025
// NAIVE multicore LU solve version

// The barrier must NOT be a plain global: globals are linked into L3 (the
// linker script only has the DRAM region), so every snrt_partial_barrier would
// spin on a DRAM address at ~60 cycles per access. It is allocated in TCDM by
// the DM core in main instead, and this global only holds the pointer.
snrt_barrier_t *barr;

void lu_solve_naive(float *mat, uint32_t *perm, float *y, float *vec, float *result,
                    float *local_sum) {


    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t ncores   = snrt_cluster_compute_core_num();
    uint32_t perm_idx;

    // Local copy: otherwise the global pointer is re-read from DRAM after
    // every barrier call (the call writes memory, so it cannot be cached)
    snrt_barrier_t *bar_p = barr;

    // -------------------------
    // FORWARD SUBSTITUTION (L * y = P * vec)
    // -------------------------
    for (uint32_t m = 0; m < N; m++) {
        
        // reset buffer
        local_sum[core_idx] = 0.0f;
    
        // numero di elementi da sommare: k in [0, m)
        uint32_t count = m; // può essere 0
        uint32_t block = count / (uint32_t)ncores;
        uint32_t left  = count % (uint32_t)ncores;

        // start/end per questo core (distribuzione dei resti sui primi 'left' core)
        uint32_t start = core_idx * block + (core_idx < (uint32_t)left ? core_idx : left);
        uint32_t end   = start + block + (core_idx < (uint32_t)left ? 1 : 0);

        for (uint32_t k = start; k < end; k++)
            local_sum[core_idx] += mat[m * N + k] * y[k];
        

        // barrier: attendi che tutti i core finiscano
        snrt_partial_barrier(bar_p, 8);

        // core 0 somma tutte le parti e aggiorna y[m]
        if (core_idx == 0) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < ncores; i++)
                sum += local_sum[i];

            perm_idx = perm[m];
            y[m] = vec[perm_idx] - sum;
        }
        
        // barrier: tutti i core attendono che y[m] sia calcolato
        snrt_partial_barrier(bar_p, 8);
    }

    // BACKWARD SUBSTITUTION (U * result = y)
    
    for (int m = (int) N - 1; m >=0; m--) {

        // la finestra su cui sommare è k in [m+1, N)
        uint32_t count = (uint32_t)N - (m + 1); // numero elementi nella finestra, può essere 0
        uint32_t block = count / (uint32_t)ncores;
        uint32_t left  = count % (uint32_t)ncores;

        uint32_t start_rel = core_idx * block + (core_idx < (uint32_t)left ? core_idx : left);
        uint32_t end_rel   = start_rel + block + (core_idx < (uint32_t)left ? 1 : 0);

        uint32_t start = (m + 1) + start_rel;
        uint32_t end   = (m + 1) + end_rel;

        snrt_partial_barrier(bar_p, 8);

        // L'azzeramento deve stare DOPO questa barriera: al giro precedente
        // core 0 legge local_sum[] dopo la seconda barriera, mentre gli altri
        // core sono già ripartiti. Azzerandolo prima della barriera un core
        // poteva cancellare la sua somma parziale mentre core 0 la stava
        // ancora sommando (nel loop in avanti non succede perché lì
        // l'azzeramento cade dopo la barriera di fine giro).
        local_sum[core_idx] = 0.0f;

        for (uint32_t k = start; k < end; k++)
            local_sum[core_idx] += mat[m * N + k] * result[k];
        

        snrt_partial_barrier(bar_p, 8);

        // core 0 somma e calcola result[m]
        if (core_idx == 0) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < ncores; i++)
                sum += local_sum[i];
            result[m] = (y[m] - sum) / mat[m * N + m];
        }
     
    }

    return;
}
