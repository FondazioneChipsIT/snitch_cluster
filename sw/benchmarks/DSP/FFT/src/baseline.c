static double *fft_base(uint32_t N, double *x, double *y, double *twiddle) {
	uint32_t core_id = snrt_cluster_core_idx();
	uint32_t core_num = snrt_cluster_compute_core_num();

    snrt_mcycle();
    // Define a generic barrier
	snrt_barrier_t *barr;
	
	for (uint32_t n = N, s = 1; n > 1; n /= 2, s *= 2) {
		uint32_t j0 = 0, js = 1, j1 = s;
		uint32_t i0 = 0, is = 1, i1 = n/2;

		if (s < core_num) {
			i0 = core_id;
			is = core_num;
			i1 = n/2 / core_num;
		} else {
			j0 = core_id;
			js = core_num;
			j1 = s / core_num;
		}
	
		for (uint32_t j = 0; j < j1; ++j) {
			// asm volatile ("loop_j_start:");
			double tw_re = twiddle[(j*js+j0)*n+0];
			double tw_im = twiddle[(j*js+j0)*n+1];
			for (uint32_t i = 0; i < i1; ++i) {
				asm volatile ("bugu:");
				double v_re = x[(i*is+i0)*s*2+(j*js+j0)*2+N+0];
				double v_im = x[(i*is+i0)*s*2+(j*js+j0)*2+N+1];
				double u_re = x[(i*is+i0)*s*2+(j*js+j0)*2+0];
				double u_im = x[(i*is+i0)*s*2+(j*js+j0)*2+1];
				double twv_re = (tw_re*v_re - tw_im*v_im);
				double twv_im = (tw_re*v_im + tw_im*v_re);
				// double twv_re;
				// double twv_im;
				double twv_re_magic;
				double twv_im_magic;
				asm volatile (
					"bubsi: \n"
					"fmul.d    ft2, %[tw_re], %[v_re] \n"
					"fmul.d    ft3, %[tw_im], %[v_im] \n"
					"fmul.d    ft4, %[tw_re], %[v_im] \n"
					"fmul.d    ft5, %[tw_im], %[v_re] \n"
					"fsub.d    %[twv_re], ft2, ft3 \n"
					"fadd.d    %[twv_im], ft4, ft5 \n"
					: [twv_re]"=f"(twv_re_magic), [twv_im]"=f"(twv_im_magic)
					: [tw_re]"f"(tw_re), [tw_im]"f"(tw_im), [v_re]"f"(v_re), [v_im]"f"(v_im)
					: "ft2", "ft3", "ft4", "ft5"
				);
				asm volatile ("gaxi:");
				double d0 = twv_re - twv_re_magic;
				double d1 = twv_im - twv_im_magic;
				if (d0 < 0) d0 = -d0;
				if (d1 < 0) d1 = -d1;
				if (d0 > 0.001 || d1 > 0.001)
					for (;;);
				asm volatile ("check_done_bro:");
				// y[(i*is+i0)*s*4+(j*js+j0)*2+0] = u_re + twv_re;
				// y[(i*is+i0)*s*4+(j*js+j0)*2+s*2+0] = u_re - twv_re;
				// y[(i*is+i0)*s*4+(j*js+j0)*2+1] = u_im + twv_im;
				// y[(i*is+i0)*s*4+(j*js+j0)*2+s*2+1] = u_im - twv_im;
				asm volatile ("gaximaxi:");
				asm volatile (
					"fmul.d   ft2, %[tw_re], %[v_re] \n"
					"fmul.d   ft3, %[tw_im], %[v_re] \n"
					"fnmsub.d ft2, %[tw_im], %[v_im], ft2 \n"
					"fmadd.d  ft3, %[tw_re], %[v_im], ft3 \n"
					"fadd.d   ft4, %[u_re], ft2 \n"
					"fsd      ft4, 0(%[y_00]) \n"
					"fsub.d   ft4, %[u_re], ft2 \n"
					"fsd      ft4, 0(%[y_10]) \n"
					"fadd.d   ft4, %[u_im], ft3 \n"
					"fsd      ft4, 0(%[y_01]) \n"
					"fsub.d   ft4, %[u_im], ft3 \n"
					"fsd      ft4, 0(%[y_11]) \n"
					:
					:
						[tw_re]"f"(tw_re),
						[tw_im]"f"(tw_im),
						[v_re]"f"(v_re),
						[v_im]"f"(v_im),
						[u_re]"f"(u_re),
						[u_im]"f"(u_im),
						[y_00]"r"(&y[(i*is+i0)*s*4+(j*js+j0)*2+0]),
						[y_10]"r"(&y[(i*is+i0)*s*4+(j*js+j0)*2+s*2+0]),
						[y_01]"r"(&y[(i*is+i0)*s*4+(j*js+j0)*2+1]),
						[y_11]"r"(&y[(i*is+i0)*s*4+(j*js+j0)*2+s*2+1])
					: "ft2", "ft3", "ft4"
				);
				asm volatile ("penil:");
			}
			// asm volatile ("loop_j_end:");
		}
		double *tmp = (double *) snrt_align_up(x, 8);
		x = (double *) snrt_align_up(y, 8);
		y = tmp;
		snrt_partial_barrier(barr, 8);
	}
    snrt_mcycle();
	return x;
}