#ifndef DATA_H
#define DATA_H

// Matrix elems (for now only a multiple of ncores, default 8)
#ifndef elems
#define elems 64
#endif

// TCDM pointers to our data
float *mat, *y, *vec, *result, *local_sum;
uint32_t *perm_vec;
// Elems * ncore vector used to write back
float *vec_write_back;

// Cycle and performance metrics
uint64_t start_cycle[16], end_cycle[16], total_cycles[16]; // Increase the num if ncores>16
float flop_cycle[16];

#endif