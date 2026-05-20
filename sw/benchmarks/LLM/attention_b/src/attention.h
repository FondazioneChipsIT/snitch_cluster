#include "snrt.h"
#include "fastexpf.h"

void attention(float* out, float* preatt, float* att,
                       float* inp, uint32_t b_s, uint32_t t_s, uint32_t c_s, uint32_t nh_s) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    uint32_t C3 = c_s*3;
    uint32_t hs = c_s / nh_s; // head size
    float scale = 1.0 / sqrtf(hs);

    // Each compute core handles a contiguous chunk of T rows
    uint32_t num_cores = snrt_cluster_compute_core_num();
    uint32_t core_id   = snrt_cluster_core_idx();
    uint32_t t_start = core_id;
    uint32_t t_step  = num_cores;

    for (uint32_t b1 = 0; b1 < b_s; b1++) {
        for (uint32_t t = t_start; t < T; t += t_step) {
            for (uint32_t h = 0; h < nh_s; h++) {
                float* query_t = inp + b1 * t_s * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b1*nh_s*t_s*t_s + h*t_s*t_s + t*t_s;
                float* att_bth = att + b1*nh_s*t_s*t_s + h*t_s*t_s + t*t_s;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (uint32_t t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b1 * t_s * C3 + t2 * C3 + h * hs + c_s; // +c_s because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (uint32_t i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (uint32_t t2 = 0; t2 <= t; t2++) {
                    float expv = fast_expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (uint32_t t2 = 0; t2 < t_s; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values uint32_to the output of attention
                float* out_bth = out + b1 * t_s * c_s + t * c_s + h * hs;
                for (uint32_t i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (uint32_t t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b1 * t_s * C3 + t2 * C3 + h * hs + c_s*2; // +c_s*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (uint32_t i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
    snrt_fpu_fence();
    return;
}