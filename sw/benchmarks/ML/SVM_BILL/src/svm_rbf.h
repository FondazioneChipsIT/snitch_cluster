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

    float d, d1, sum = 0.0f;

    // It doesn't make sense to use the SSR for this kernel computation, 
    // as the number of features is small (f_dim = 4 for BILL) and the overhead of setting up the SSR outweighs the benefits.

    for (int i = 0; i < f_dim; i += 2) {
        d  = x[i]   - y[i];
        d1 = x[i+1] - y[i+1];
        sum += d*d + d1*d1;
    }
    float gs = -gamma * sum;
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

        for (int k = 0; k < COEF_DIM; k++) {  //
            ptrs = &x_ref[k * f_dim];
            temp = rbf(ptrx, ptrs, gamma1, f_dim);  // K(xᵢ, sv_k)
            inter += temp * sv_coef[k];             // αₖ · K(...)
        }
        
        Pred[i] = (inter + bias[0] >= 0) ? 1 : 0;
    }
    snrt_fpu_fence();
    return;
}
