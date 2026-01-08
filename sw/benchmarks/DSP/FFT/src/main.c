/* Original code from the paper by F. Zaruba, ported to new functions and 
   runtime library.
   Luca Colombo, 2026, Chips-IT*/

#include "snrt.h"
#include "ssr_frep.c"
#include "data.c"

extern uint32_t input_size;
extern double input[];
extern double input_twiddle[];
extern double output[];
extern double *local_x;
extern double *local_tw;

static void populate(double *ptr, uint32_t size, uint32_t seed) {
    for (uint32_t i = 0; i < size; i++) {
        *ptr = (double)seed * 3.141;
        ++ptr;
        ++seed;
    }
}

int main() {

	uint32_t core_id = snrt_cluster_core_idx();

	 // Copy data in TCDM
    if (snrt_is_dm_core()) {
		// Generate addresses in L1
		local_x = (double *)snrt_l1_next();
		local_tw = local_x + input_size*2;
 		populate(local_x, input_size*2, 1);
    	populate(local_tw, input_size, 2);
		printf("Populated!\n");
    }

	snrt_cluster_hw_barrier();

	double *y;
	// We allocate buffer already in l1
	double *buffer = local_tw + input_size;

	if(snrt_is_compute_core()){
		printf("Core %d: %p | %p | %p \n", core_id, (void*)local_x, (void*)local_tw, (void*)buffer);//stampa i puntatori per vedere c he cazzo succede 

		y = fft_inner(input_size, local_x, buffer, local_tw, 1);
	}

	snrt_cluster_hw_barrier();

	if (core_id == 0) {
		uint32_t diffs = 0;
		for (uint32_t i = 0; i < input_size*2; i++) {
			double d = y[i] - output[i];
			if (d < 0)
				d = -d;
			diffs += d > 0.01;
		}
		return diffs;
	}

	return 0;
}
