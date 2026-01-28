/* Original code from the paper by F. Zaruba, ported to new functions and 
   runtime library.
   Luca Colombo, 2026, Chips-IT*/

#include "snrt.h"
#include "ssr_frep.c"
#include "baseline.c"
#include "data.h"

// Global pointers to L1
double *local_tw, *local_x, *buffer;

// Multicore or single core
bool multi_core = 1;

// Optimized or baseline
bool use_opt = 1;

int main() {

	uint32_t core_id = snrt_cluster_core_idx();

	// Copy data in TCDM
    if (snrt_is_dm_core()) {
		// Generate addresses in L1
		local_x  = (double *) snrt_l1_next();
		local_tw = local_x + FFT_N*2;
		buffer = local_tw + FFT_N;

		size_t size1 = FFT_N*2 * sizeof(double);
		size_t size2 = FFT_N * sizeof(double);

        snrt_dma_start_1d(local_x, input, size1);
        snrt_dma_start_1d(local_tw, twiddle, size2);
        snrt_dma_wait_all();
    }

	snrt_cluster_hw_barrier();
	
	double *y;
    
	if(snrt_is_compute_core()){

		if(use_opt)
			y = fft_inner(FFT_N, local_x, buffer, local_tw);
		else	
			y = fft_base(FFT_N, local_x, buffer, local_tw);

	}

	snrt_cluster_hw_barrier();

	// Check against golden model
	if (core_id == 0) {
		uint32_t diffs = 0;
		for (uint32_t i = 0; i < FFT_N; i++) {
			double d = y[i] - output[i];
			if (d < 0){
				d = -d;
				
			}
			//printf("Index %d: Computed %f, Golden %f, Diff %f\n", i, y[i], output[i], d);
			diffs += d > 0.01;
		}
		return diffs;
	}

	return 0;
}