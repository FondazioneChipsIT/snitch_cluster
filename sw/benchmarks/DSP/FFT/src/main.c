/* Original code from the paper by F. Zaruba, ported to new functions and 
   runtime library.
   Luca Colombo, 2026, Chips-IT*/

#include "snrt.h"
#include "ssr_frep.c"
#include "baseline.c"
#include "data.h"

// Global pointers to L1
double *local_tw, *local_x;

// Multicore or single core
bool multi_core = 1;

// Optimized or baseline
bool use_opt = 1;

int main() {

	uint32_t core_id = snrt_cluster_core_idx();

	// Copy data in TCDM
    if (snrt_is_dm_core()) {
		// Generate addresses in L1
		local_x  = (double *)snrt_align_up(snrt_l1_next(), 8);
		local_tw = (double *)snrt_align_up(local_x + FFT_N*2, 8);

		size_t size = FFT_N*2 * sizeof(double);

        snrt_dma_start_1d(local_x, input, size);
        snrt_dma_start_1d(local_tw, twiddle, size);
        snrt_dma_wait_all();
    }

	snrt_cluster_hw_barrier();
	// We allocate buffer already in l1
	double *y = (double *)snrt_align_up(local_tw + FFT_N*2, 8);
	
	if(snrt_is_compute_core()){

		if(multi_core){
			if(use_opt)
				fft_inner(FFT_N, local_x, y, local_tw, multi_core);
			else	
				fft_base(FFT_N, local_x, y, local_tw, multi_core);
		}
		else
			if(core_id==0){
				if(use_opt)
					fft_inner(FFT_N, local_x, y, local_tw, multi_core);
				else	
					fft_base(FFT_N, local_x, y, local_tw, multi_core);
			}
	}

	snrt_cluster_hw_barrier();

	// Check against golden model
	if (core_id == 0) {
		uint32_t diffs = 0;
		for (uint32_t i = 0; i < FFT_N; i++) {
			double d = y[i] - golden[i];
			if (d < 0)
				d = -d;
			diffs += d > 0.01;
		}
		return diffs;
	}

	return 0;
}
