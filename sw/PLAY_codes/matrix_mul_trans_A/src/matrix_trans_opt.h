// Luca Colombo Chips-IT 2025
/* We access src matrix by rows, moving element by element
   then we access dst by columns with a stride equal to a multiple of rows (matrix in L1 is placed row after row).
   We copy each src row in the dst column: example 16x16 matrix:
   core 0 will read the (j+offset)*elems row of src, copy in the j+offset column of dst, where:
   offset is the core offset, for 0 is 0, for 1 is 1*chunk_per_core etc.
   j starts as 0 and is incremented to be < chunk_per_core (elems/ncores), so in this case 2 (each core processes 2 rows/cols)
   for the rows we multiply j+offset times elems to move from a row to the next
*/
void matrix_trans_opt   (uint32_t chunk_per_core, uint32_t offset,
                         double *src, double *dst) {

    double zero = 0.0; // Zero register

    /* Load zero into ft3 */
    asm volatile(
        "fld ft3, 0(%[zero])\n"
        :
        : [zero] "r"(&zero)
        : "ft3");

    // Read the src matrix with a 1 stride-> access a row (chunk*elemns to read more rows)
    snrt_ssr_loop_1d(SNRT_SSR_DM0, chunk_per_core * elems, sizeof(double));
    // Write the dst matrix with a elems stride-> access a column (in memory the matrix is put
    // row after row)
    snrt_ssr_loop_1d(SNRT_SSR_DM1, elems, elems*sizeof(double)); 

    // To select the src row, (offset)*elems, where offset depends on the core
    // It is possible to call this out of the loop as the rows are contigous
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, src + offset*(size_t)elems);
    
    for(uint32_t j = 0; j < chunk_per_core; j++){

        // As for the dst column the write is easy as it is only offset+j (stored as rows)
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, dst + j + offset);
        
        snrt_ssr_enable();

        /* Repeat rows times: ft1 = ft0 + ft2 (0)*/
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"
            "fadd.d ft1, ft0, ft3\n"
            :
            : [n_frep] "r"(elems - 1)
            : "ft0", "ft1", "ft3", "memory");

        snrt_ssr_disable();
        
        snrt_fpu_fence();   
        }


    return;
}