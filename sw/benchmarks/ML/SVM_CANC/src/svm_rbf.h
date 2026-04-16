// Luca Colombo Chips-IT 2026
#include "snrt.h"

// Values used to perform the fast exponential approximation, based on the GIST algorithm
#define GIST_A 12102203.17133801f
#define GIST_B 1064986823.010288f
#define GIST_C 8388608
#define GIST_D 2139095040

// GIST-based fast exponential approximation, used to compute the RBF kernel
float fastexp_gist(float x)
{
    x = GIST_A * x + GIST_B;

    if (x < GIST_C || x > GIST_D)
        x = (x < GIST_C) ? 0.0f : GIST_D;

    uint32_t n = (uint32_t)(x);
    return *(float *)&n;
}

// RBF kernel computation, used in the SVM algorithm
float rbf(float *x, float *y, float gamma, int f_dim){
    
    float gs;
    float gamma_n = -gamma; 
    float zero = 0.0f;
    asm volatile(
    "flw ft5, 0(%[zero])\n" // accumulator1
    "flw ft6, 0(%[zero])\n" // accumulator2
    "flw ft7, 0(%[gamma_n])\n" 
    :
    : [zero] "r"(&zero), [gamma_n] "r"(&gamma_n)
    : "ft5", "ft6", "ft7", "memory");

    snrt_ssr_loop_1d(SNRT_SSR_DM0, f_dim, sizeof(float));
    snrt_ssr_loop_1d(SNRT_SSR_DM1, f_dim, sizeof(float));
    // read x and y, write gs
    snrt_ssr_read(SNRT_SSR_DM0,SNRT_SSR_1D, x);
    snrt_ssr_read(SNRT_SSR_DM1,SNRT_SSR_1D, y); 

    snrt_ssr_enable();
    asm volatile(
        "frep.o %[n_frep], 4, 0, 0 \n"  
        "fsub.s ft3, ft0, ft1\n" // Unroll by 2, first subtract the elements
        "fsub.s ft4, ft0, ft1\n"
        "fmadd.s ft5, ft3, ft3, ft5\n" // Then square the differences and accumulate the sum of the squared difference
        "fmadd.s ft6, ft4, ft4, ft6\n"
        // Sum of the squared differences and multiply by gamma
        "fadd.s ft6, ft5, ft6\n" // Sum the two accumulators
        "fmul.s ft6, ft6, ft7\n" // Multiply by gamma
        "fsw ft6, 0(%[gs])\n"    // store the result in gs
        : 
        : [n_frep] "r"(f_dim/2 - 1), [gs] "r"(&gs)
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "memory");

    snrt_ssr_disable();
    /* Shoul be equivalent to this
    for (int i = 0; i < f_dim; i += 2) {
        d  = x[i]   - y[i];
        d1 = x[i+1] - y[i+1];
        sum += d*d + d1*d1;
    }*/

    return fastexp_gist(gs); 
}


void SVM_RBF(uint32_t core_idx, uint32_t chunk_per_core, uint32_t offset, 
            float *data_model, float *Pred, float *x_ref, 
            float bias[1], float* sv_coef, float gamma1, int f_dim) {
    float inter, temp;
    float *ptrx, *ptrs;

    for (int i = offset; i < offset + chunk_per_core; i++) {
        inter = 0.0f;
        ptrx = &data_model[i * f_dim];

        for (int k = 0; k < COEF_DIM; k++) {  // coef_dim = 100 support vectors
            ptrs = &x_ref[k * f_dim];
            temp = rbf(ptrx, ptrs, gamma1, f_dim);  // K(xᵢ, sv_k)
            inter += temp * sv_coef[k];             // αₖ · K(...)
        }
        
        Pred[i] = (inter + bias[0] >= 0) ? 1 : 0;
    }

    return;
}
