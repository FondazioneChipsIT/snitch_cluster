/* Original code from the paper by F. Zaruba, ported to new functions and 
   runtime library.
   Luca Colombo, 2026, Chips-IT*/

#include "snrt.h"
#include "ssr_frep.c"
#include "baseline.c"
#include "data.h"

// Global pointers to L1
float *local_tw, *local_x, *buffer;

// Multicore or single core
bool multi_core = 1;

// Optimized or baseline
bool use_opt = 1;

int main() {

	uint32_t core_id = snrt_cluster_core_idx();

	// Copy data in TCDM
    if (snrt_is_dm_core()) {
		// Generate addresses in L1
		local_x  = (float *) snrt_l1_next();
		local_tw = local_x + FFT_N*2;
		buffer = local_tw + FFT_N;

		size_t size1 = FFT_N*2 * sizeof(float);
		size_t size2 = FFT_N * sizeof(float);

        snrt_dma_start_1d(local_x, input, size1);
        snrt_dma_start_1d(local_tw, twiddle, size2);
        snrt_dma_wait_all();
    }

	float *y;

	snrt_cluster_hw_barrier();
	snrt_mcycle(); // Start cycle count
	
	if(snrt_is_compute_core()){

		if(use_opt)
			y = fft_inner(FFT_N, local_x, buffer, local_tw);
		else	
			y = fft_base(FFT_N, local_x, buffer, local_tw);

	}

	snrt_cluster_hw_barrier();
	snrt_mcycle();

	// Check against golden model
	if (core_id == 0) {
		float eps = 0.01f;
		uint32_t diffs = 0;
		for (uint32_t i = 0; i < FFT_N; i++) {
			float d = y[i] - output[i];
			d = fabsf(d);
			//printf("Index %d: Computed %f, Golden %f, Diff %f\n", i, y[i], output[i], d);
			diffs += d > eps;
		}
		return diffs;
	}

	return 0;
}