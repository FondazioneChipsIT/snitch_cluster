static float *fft_base(uint32_t N, float *x, float *y, float *twiddle) {
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
			float tw_re = twiddle[(j*js+j0)*n+0];
			float tw_im = twiddle[(j*js+j0)*n+1];
			for (uint32_t i = 0; i < i1; ++i) {
				asm volatile ("bugu:");
				float v_re = x[(i*is+i0)*s*2+(j*js+j0)*2+N+0];
				float v_im = x[(i*is+i0)*s*2+(j*js+j0)*2+N+1];
				float u_re = x[(i*is+i0)*s*2+(j*js+j0)*2+0];
				float u_im = x[(i*is+i0)*s*2+(j*js+j0)*2+1];
				float twv_re = (tw_re*v_re - tw_im*v_im);
				float twv_im = (tw_re*v_im + tw_im*v_re);
				// float twv_re;
				// float twv_im;
				float twv_re_magic;
				float twv_im_magic;
				asm volatile (
					"bubsi: \n"
					"fmul.s    ft2, %[tw_re], %[v_re] \n"
					"fmul.s    ft3, %[tw_im], %[v_im] \n"
					"fmul.s    ft4, %[tw_re], %[v_im] \n"
					"fmul.s    ft5, %[tw_im], %[v_re] \n"
					"fsub.s    %[twv_re], ft2, ft3 \n"
					"fadd.s    %[twv_im], ft4, ft5 \n"
					: [twv_re]"=f"(twv_re_magic), [twv_im]"=f"(twv_im_magic)
					: [tw_re]"f"(tw_re), [tw_im]"f"(tw_im), [v_re]"f"(v_re), [v_im]"f"(v_im)
					: "ft2", "ft3", "ft4", "ft5"
				);
				asm volatile ("gaxi:");
				float d0 = twv_re - twv_re_magic;
				float d1 = twv_im - twv_im_magic;
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
					"fmul.s   ft2, %[tw_re], %[v_re] \n"
					"fmul.s   ft3, %[tw_im], %[v_re] \n"
					"fnmsub.s ft2, %[tw_im], %[v_im], ft2 \n"
					"fmadd.s  ft3, %[tw_re], %[v_im], ft3 \n"
					"fadd.s   ft4, %[u_re], ft2 \n"
					"fsw      ft4, 0(%[y_00]) \n"
					"fsub.s   ft4, %[u_re], ft2 \n"
					"fsw      ft4, 0(%[y_10]) \n"
					"fadd.s   ft4, %[u_im], ft3 \n"
					"fsw      ft4, 0(%[y_01]) \n"
					"fsub.s   ft4, %[u_im], ft3 \n"
					"fsw      ft4, 0(%[y_11]) \n"
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
		float *tmp = (float *) snrt_align_up(x, 8);
		x = (float *) snrt_align_up(y, 8);
		y = tmp;
		snrt_partial_barrier(barr, 8);
	}
    snrt_mcycle();
	return x;
}