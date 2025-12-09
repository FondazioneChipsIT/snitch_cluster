/* Verification: compute PAorig and L*U and Frobenius norm of difference.
   perm_vec: perm_vec[i] = original row index now at position i
*/

static double snitch_sqrt(double x) {
    if (x <= 0.0) return 0.0;

    double y = x;
    // 57 iterations are enough for double precision
    for (int i = 0; i < 7; i++) {
        y = 0.5 * (y + x / y);
    }
    return y;
}

void verify_lu(double *Aorig, double *mat, int *perm, uint32_t n) {
    double PA[elems*elems];
    double LU[elems*elems];
    /* Build PA */
    for (uint32_t i=0;i<n;i++){
        int orig = perm[i];
        for (uint32_t j=0;j<n;j++) PA[i*n + j] = Aorig[orig*n + j];
    }
    /* Build L and U and compute LU = L*U */
    for (uint32_t i=0;i<n;i++) for (uint32_t j=0;j<n;j++) LU[i*n+j] = 0.0;
    for (uint32_t i=0;i<n;i++){
        for (uint32_t k=0;k<n;k++){
            double Lik = (k < i) ? mat[i*n + k] : (k==i ? 1.0 : 0.0);
            if (Lik == 0.0) continue;
            for (uint32_t j=k;j<n;j++){
                double Ukj = mat[k*n + j];
                LU[i*n + j] += Lik * Ukj;
            }
        }
    }
    /* compute Frobenius norm of PA-LU and ||PA|| */
    double err2 = 0.0, norm2 = 0.0;
    for (uint32_t i=0;i<n*n;i++){
        double d = PA[i] - LU[i];
        err2 += d*d;
        norm2 += PA[i]*PA[i];
    }
    double err = snitch_sqrt(err2);
    double norm = snitch_sqrt(norm2);
    double rel = (norm>0.0) ? err / norm : err;
    printf("VERIFY: ||PA - L*U||_F = %g, ||PA||_F = %g, rel = %g\n", err, norm, rel);
    return ;
}