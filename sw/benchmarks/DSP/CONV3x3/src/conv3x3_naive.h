// Luca Colombo Chips-IT 2025
/* Naive Conv3x3 matrix, same logical traversal of conv3x3_opt
   but implemented with plain C loops (no SSR, no asm) */

void conv3x3_naive(uint32_t core_idx, uint32_t chunk_per_core, uint32_t offset,
                   float *x, float *y, float *h){
    snrt_mcycle();
    if (core_idx < 6 ||
        (chunk_per_core > 1 && core_idx == 6) ||
        (chunk_per_core > 2 && core_idx == 7)) {

        uint32_t tot_rows = chunk_per_core;
        if (core_idx == 7) {
            tot_rows = chunk_per_core - 2;
        }


        for (uint32_t r = 0; r < tot_rows; r++) {

            uint32_t x_row = offset + r;
            uint32_t y_row = offset + r;


            for (uint32_t c = 0; c < LEN - 2; c++) {

                float acc = 0.0f;

                // Finestra 3x3
                for (uint32_t kr = 0; kr < 3; kr++) {
                    for (uint32_t kc = 0; kc < 3; kc++) {

                        float x_val =
                            x[(x_row + kr) * LEN + (c + kc)];
                        float h_val =
                            h[kr * 3 + kc];

                        acc += x_val * h_val;
                    }
                }


                y[y_row * (LEN - 2) + c] = acc;
            }
        }
        snrt_mcycle();
    }
    
}
