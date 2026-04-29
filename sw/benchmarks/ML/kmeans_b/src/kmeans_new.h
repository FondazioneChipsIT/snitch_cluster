#include "math.h"
#include "snrt.h"

float euclidean_distance(uint32_t n_features, float* p1, float* p2) {
    float sum = 0.0f;
    //for (uint32_t i = 0; i < n_features; i++) {
    //    float diff = p1[i] - p2[i];
    //    sum += diff * diff;
    //}
    
    float zero = 0.0f;
    asm volatile(
    "flw ft5, 0(%[zero])\n" // accumulator1
    "flw ft6, 0(%[zero])\n" // accumulator2 
    :
    : [zero] "r"(&zero)
    : "ft5", "ft6", "memory");
    snrt_ssr_loop_1d(SNRT_SSR_DM0, n_features, sizeof(float));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, n_features, sizeof(float));
    // read x and y, write gs
    snrt_ssr_read(SNRT_SSR_DM0,SNRT_SSR_1D, p1);
    snrt_ssr_read(SNRT_SSR_DM1,SNRT_SSR_1D, p2); 
    snrt_fpu_fence();
    snrt_ssr_enable();
    asm volatile(

        "frep.o %[n_frep], 4, 0, 0 \n"  
        "fsub.s ft3, ft0, ft1\n" 
        "fsub.s ft4, ft0, ft1\n"
        "fmadd.s ft5, ft3, ft3, ft5\n" // accumulate the sum of the squared difference
        "fmadd.s ft6, ft4, ft4, ft6\n"
        // Sum of the squared differences and multiply by gamma
        "fadd.s ft6, ft5, ft6\n" // Sum the two accumulators
        "fsw ft6, 0(%[sum])\n"    // store the result in sum
        : 
        : [n_frep] "r"(n_features/2 - 1), [sum] "r"(&sum)
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "memory");
    snrt_ssr_disable();

    return sum;
}

snrt_barrier_t barr;

void kmeans_iteration(
    uint32_t  start_idx,   
    uint32_t  end_idx,      
    uint32_t  n_clusters,
    uint32_t  n_features,
    float*    samples,      // tutti i campioni in TCDM
    uint32_t* membership,   // array assegnazioni [n_samples]
    uint32_t* my_cnt,       // contatori parziali di QUESTO core [n_clusters]
    float*    centroids,    // centroidi correnti [n_clusters * n_features] ← aggiornati in-place
    float*    my_partial,   // somme parziali di QUESTO core [n_clusters * n_features]
    uint32_t* all_cnt,      // base di tutti i contatori [n_cores * n_clusters]
    float*    all_partial   // base di tutte le somme    [n_cores * n_clusters * n_features]
) {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t n_cores  = snrt_cluster_compute_core_num();

    
    for (uint32_t k = 0; k < n_clusters; k++) {
        my_cnt[k] = 0;
        for (uint32_t f = 0; f < n_features; f++)
            my_partial[k * n_features + f] = 0.0f;
    }

   
    for (uint32_t si = start_idx; si < end_idx; si++) {
        float*   samp     = &samples[si * n_features];
        float    min_dist = __builtin_inff();
        uint32_t best_k   = 0;

        for (uint32_t k = 0; k < n_clusters; k++) {
            float d = euclidean_distance(n_features, samp, &centroids[k * n_features]);
            if (d < min_dist) { min_dist = d; best_k = k; }
        }

        membership[si] = best_k;
        my_cnt[best_k]++;
        for (uint32_t f = 0; f < n_features; f++)
            my_partial[best_k * n_features + f] += samp[f];
    }

    // Tutti i core hanno finito di scrivere le proprie fette
    snrt_partial_barrier(&barr, 8);

    // ── Update step: solo core 0 riduce e normalizza ─────────────────────────
    if (core_idx == 0) {
        // Accumula le fette dei core 1..n_cores-1 nella fetta 0
        for (uint32_t c = 1; c < n_cores; c++) {
            uint32_t* remote_cnt  = all_cnt     + c * n_clusters;
            float*    remote_part = all_partial  + c * n_clusters * n_features;

            for (uint32_t k = 0; k < n_clusters; k++) {
                all_cnt[k] += remote_cnt[k];
                for (uint32_t f = 0; f < n_features; f++)
                    all_partial[k * n_features + f] +=
                        remote_part[k * n_features + f];
            }
        }

        // Normalizza -> nuovi centroidi scritti in-place
        for (uint32_t k = 0; k < n_clusters; k++) {
            uint32_t cnt = all_cnt[k];
            if (cnt == 0) continue; // centroide vuoto: lascia invariato
            for (uint32_t f = 0; f < n_features; f++)
                centroids[k * n_features + f] =
                    all_partial[k * n_features + f] / (float)cnt;
        }
    }

    snrt_partial_barrier(&barr, 8);
    return;
}