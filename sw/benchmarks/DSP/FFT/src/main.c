/* Original code from the paper by F. Zaruba, ported to new functions and 
   runtime library.
   Luca Colombo, 2026, Chips-IT*/

#include "snrt.h"
#include "ssr_frep.c"
#include "baseline.c"
#include "data.c"

extern uint32_t input_size;
extern double input[];
extern double input_twiddle[];
extern double output[];
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
		local_x = (double *)snrt_l1_next();
		local_tw = local_x + input_size;

		size_t size = input_size * sizeof(double);

        snrt_dma_start_1d(local_x, input, size);
        snrt_dma_start_1d(local_tw, input_twiddle, size);
        snrt_dma_wait_all();
    }

	snrt_cluster_hw_barrier();
	// We allocate buffer already in l1
	double *y = local_tw + input_size;

	if(snrt_is_compute_core()){

		if(multi_core){
			if(use_opt)
				fft_inner(input_size, local_x, y, local_tw, multi_core);
			else	
				fft_base(input_size, local_x, y, local_tw, multi_core);
		}
		else
			if(core_id==0){
				if(use_opt)
					fft_inner(input_size, local_x, y, local_tw, multi_core);
				else	
					fft_base(input_size, local_x, y, local_tw, multi_core);
			}
	}

	snrt_cluster_hw_barrier();

	/*if (core_id == 0) {
		uint32_t diffs = 0;
		for (uint32_t i = 0; i < input_size; i++) {
			double d = y[i] - output[i];
			if (d < 0)
				d = -d;
			diffs += d > 0.01;
		}
		return diffs;
	}*/

	return 0;
}
