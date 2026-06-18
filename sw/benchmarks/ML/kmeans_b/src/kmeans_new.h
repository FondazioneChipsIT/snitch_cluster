#include "math.h"
#include "snrt.h"

float euclidean_distance(uint32_t n_features, float* p1, float* p2) {
    float sum = 0.0f;
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
    snrt_fpu_fence();
    
    return sum;
}

snrt_barrier_t barr;

void kmeans_iteration(
    uint32_t  start_idx,   
    uint32_t  end_idx,      
    uint32_t  n_clusters,
    uint32_t  n_features,
    float*    samples,      
    uint32_t* membership,   
    uint32_t* local_newClusterSize,       
    float*    centroids,    
    float*    local_newClusters,   
    uint32_t* all_cnt,      
    float*    all_partial,   
    uint32_t  max_iter,
    uint32_t  n_samples,
    float* local_delta
) {
    uint32_t i, j, k, index; // Nomi variabili come in Codice 2
    uint32_t core_id = snrt_cluster_core_idx();
    uint32_t num_cores = snrt_cluster_compute_core_num();
    
    uint32_t iter = 0;

    snrt_partial_barrier(&barr, 8);

    // Calcolo blocchi clusters (come Codice 2)
    const uint32_t blockSize2 = (n_clusters + num_cores - 1) / num_cores;
    const uint32_t start2     = core_id * blockSize2;
    uint32_t end2             = start2 + blockSize2;
    if (end2 > n_clusters) end2 = n_clusters;

    // Reset locale prima di iniziare il calcolo
    for (j = 0; j < n_clusters; j++) {
        local_newClusterSize[j] = 0;
        for (k = 0; k < n_features; k++)
            local_newClusters[j * n_features + k] = 0.0f;
    }

    do
    {
        local_delta[core_id] = 0.0f;

        for (i = start_idx; i < end_idx; i++)
        {
            
            float* samp = &samples[i * n_features];
            float min_dist = __builtin_inff();
            index = 0;
            
            for (k = 0; k < n_clusters; k++) {
                float dist = euclidean_distance(n_features, samp, &centroids[k * n_features]);
                if (dist < min_dist) {
                    min_dist = dist;
                    index = k;
                }
            }

            if (membership[i] != index)
                local_delta[core_id] += 1.0f;

            membership[i] = index;

            local_newClusterSize[index]++;
            for (j = 0; j < n_features; j++)
                local_newClusters[index * n_features + j] += samp[j];
        }

        snrt_partial_barrier(&barr, 8); // pi_cl_team_barrier() 1

        if (core_id == 0)
        {
            
            for (j = 1; j < num_cores; j++)
                local_delta[0] += local_delta[j];
        }


        for (i = start2; i < end2; i++)
        {
            uint32_t global_sz = 0;
            for (j = 0; j < num_cores; j++)
            {
                global_sz += all_cnt[j * n_clusters + i];
                all_cnt[j * n_clusters + i] = 0;
            }
            all_cnt[i] = global_sz; 

            for (k = 0; k < n_features; k++)
            {
                float global_feat = 0.0f;
                for (j = 0; j < num_cores; j++)
                {
                    global_feat += all_partial[(j * n_clusters + i) * n_features + k];
                    all_partial[(j * n_clusters + i) * n_features + k] = 0.0f;
                }
                all_partial[i * n_features + k] = global_feat;
            }
        }

        snrt_partial_barrier(&barr, 8);

       
        if (core_id == 0)
        {
            for (i = 0; i < n_clusters; i++)
            {
                for (j = 0; j < n_features; j++)
                {
                    if (all_cnt[i] > 0)
                        centroids[i * n_features + j] = all_partial[i * n_features + j] / (float)all_cnt[i];

                    all_partial[i * n_features + j] = 0.0f;
                }
                all_cnt[i] = 0;
            }

            local_delta[0] /= (float)n_samples;

            for (j = 1; j < num_cores; j++)
                local_delta[j] = local_delta[0];

            iter++;
        }

        snrt_partial_barrier(&barr, 8); 

    } while (local_delta[core_id] > threshold && iter < max_iter); 

    if(core_id == 0)
        printf("Converged in %u iterations\n", iter);

    return;
}