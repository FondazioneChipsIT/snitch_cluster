/* Verification: compute PAorig and L*U and Frobenius norm of difference.
   perm_vec: perm_vec[i] = original row index now at position i
*/

static float snitch_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;

    float y = x;
    // 57 iterations are enough for float precision
    for (int i = 0; i < 7; i++) {
        y = 0.5f * (y + x / y);
    }
    return y;
}

void verify_lu(float *Aorig, float *mat, int *perm, uint32_t n) {
    float PA[rows*cols];
    float LU[cols*cols];
    /* Build PA */
    for (uint32_t i=0;i<n;i++){
        int orig = perm[i];
        for (uint32_t j=0;j<n;j++) PA[i*n + j] = Aorig[orig*n + j];
    }
    /* Build L and U and compute LU = L*U */
    for (uint32_t i=0;i<n;i++) for (uint32_t j=0;j<n;j++) LU[i*n+j] = 0.0f;
    for (uint32_t i=0;i<n;i++){
        for (uint32_t k=0;k<n;k++){
            float Lik = (k < i) ? mat[i*n + k] : (k==i ? 1.0f : 0.0f);
            if (Lik == 0.0f) continue;
            for (uint32_t j=k;j<n;j++){
                float Ukj = mat[k*n + j];
                LU[i*n + j] += Lik * Ukj;
            }
        }
    }
    /* compute Frobenius norm of PA-LU and ||PA|| */
    float err2 = 0.0f, norm2 = 0.0f;
    for (uint32_t i=0;i<n*n;i++){
        float d = PA[i] - LU[i];
        err2 += d*d;
        norm2 += PA[i]*PA[i];
    }
    float err = snitch_sqrt(err2);
    float norm = snitch_sqrt(norm2);
    float rel = (norm>0.0f) ? err / norm : err;
    // printf("VERIFY: ||PA - L*U||_F = %g, ||PA||_F = %g, rel = %g\n", err, norm, rel);
    return ;
}