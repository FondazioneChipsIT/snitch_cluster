/* Original code from the paper by F. Zaruba, ported to new functions and 
   runtime library.
   Luca Colombo, 2026, Chips-IT*/

static void fft_inner(uint32_t N, double *x, double *y, double *twiddle, bool multi_core) {
	uint32_t core_id = snrt_cluster_core_idx();
	uint32_t core_num = snrt_cluster_compute_core_num();
	
	snrt_mcycle();
	snrt_ssr_enable();

	for (uint32_t n = N, s = 1; n > 1; n /= 2, s *= 2) {
		uint32_t j0 = 0, js = 1, j1 = s;
		uint32_t i0 = 0, is = 1, i1 = n/2;
		if (multi_core) {
			if (s < core_num) {
				i0 = core_id;
				is = core_num;
				i1 = n/2 / core_num;
			} else {
				j0 = core_id;
				js = core_num;
				j1 = s / core_num;
			}
		}

		snrt_ssr_loop_4d(SNRT_SSR_DM0, 2, 2, i1, j1, 
			sizeof(double), -sizeof(double) * N, sizeof(double) * is * s * 2, sizeof(double) * js * 2);

		snrt_ssr_repeat(SNRT_SSR_DM0,2);

		snrt_ssr_loop_4d(SNRT_SSR_DM1, 2, 2, i1, j1, 
			sizeof(double) * s * 2, sizeof(double), sizeof(double) * is * s * 4, sizeof(double) * js * 2);

		snrt_ssr_read (SNRT_SSR_DM0, SNRT_SSR_4D, &x[i0*s*2+j0*2+N]);
		snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_4D, &y[i0*s*4+j0*2]);

		for (uint32_t j = 0; j < j1; ++j) {

			// Force the loads to avoid compiler using ft2
			asm volatile(
            "fld ft5, 0(%[tw_re])\n"
            "fld ft6, 0(%[tw_im])\n"
            :
            : [tw_re] "r"(&twiddle[(j*js+j0)*n+0]), [tw_im] "r"(&twiddle[(j*js+j0)*n+1])
            : "ft5", "ft6", "memory");

			asm volatile (
                "frep.o %[n_frep], 8, 0, 0\n"
				"fmul.d    ft4, ft5, ft0 \n"
				"fmul.d    ft3, ft6, ft0 \n"
				"fnmsub.d  ft4, ft6, ft0, ft4 \n"
				"fmadd.d   ft3, ft5, ft0, ft3 \n"
				"fadd.d    ft1, ft0, ft4 \n"
				"fsub.d    ft1, ft0, ft4 \n"
				"fadd.d    ft1, ft0, ft3 \n"
				"fsub.d    ft1, ft0, ft3 \n"
				:
				: [n_frep] "r"(i1-1)
				: "ft0", "ft1", "ft3", "ft4", "ft5", "ft6", "memory");
		}

		// Synchronize and swap buffers.
		// Need to use align or it will give misaligned stores
		double *tmp = (double *) snrt_align_up(x, 8);
		x = (double *) snrt_align_up(y, 8);
		y = tmp;
	
	}

	// Storeback
	if(core_id == 0)
		y = x;

	snrt_ssr_disable();
	snrt_mcycle();

	return;
}
