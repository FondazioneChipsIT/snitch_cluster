#include "snrt.h"
#include "fastexpf.h"

void attention_forward(float *out, float *preatt, float *att,
                                 float *inp,
                                 uint32_t B, uint32_t T, uint32_t C, uint32_t NH) {
    uint32_t C3        = C * 3;
    uint32_t hs        = C / NH;
    float scale   = 1.0f / sqrtf((float)hs);
    uint32_t total     = B * T * NH;         

    uint32_t core_idx  = snrt_cluster_core_idx();
    uint32_t num_cores = snrt_cluster_compute_core_num();

    for (uint32_t idx = core_idx; idx < total; idx += num_cores) {

        // srotola idx → (b, t, h)
        uint32_t h = idx % NH;
        uint32_t t = (idx / NH) % T;
        uint32_t b =  idx / (NH * T);

        float *query_t    = inp + b * T * C3 + t * C3 + h * hs;
        float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
        float *att_bth    = att    + b * NH * T * T + h * T * T + t * T;

        // pass 1: dot product + maxval
        float maxval = -10000.0f;
        for (uint32_t t2 = 0; t2 <= t; t2++) {
            float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
            float val = 0.0f;
            for (uint32_t i = 0; i < hs; i++)
                val += query_t[i] * key_t2[i];
            val *= scale;
            if (val > maxval) maxval = val;
            preatt_bth[t2] = val;
        }

        // pass 2: exp + sum
        float expsum = 0.0f;
        for (uint32_t t2 = 0; t2 <= t; t2++) {
            float expv = fast_expf(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // pass 3: softmax + causal mask
        for (uint32_t t2 = 0; t2 < T; t2++) {
            att_bth[t2] = (t2 <= t) ? att_bth[t2] * expsum_inv : 0.0f;
        }
        
        // pass 4: weighted sum dei value
        float *out_bth = out + b * T * C + t * C + h * hs;
        for (uint32_t i = 0; i < hs; i++) out_bth[i] = 0.0f;
        for (uint32_t t2 = 0; t2 <= t; t2++) {
            float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
            float att_btht2 = att_bth[t2];
            for (uint32_t i = 0; i < hs; i++)
                out_bth[i] += att_btht2 * value_t2[i];
        }
    }
}