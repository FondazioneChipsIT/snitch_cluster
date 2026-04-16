#ifndef DATA_H
#define DATA_H

// Matrix elems (for now only a multiple of ncores, default 8)
#ifndef rows
#define rows 64
#endif
#ifndef cols
#define cols 32
#endif


// TCDM pointers to our data
float *mat, *orig_buf, *row_k, *row_b;
int *perm_vec;
// Elems * ncore vector used to write back
float *vec_write_back;

// Cycle and performance metrics
uint64_t start_cycle[16], end_cycle[16], total_cycles[16]; // Increase the num if ncores>16
float flop_cycle[16];
// How many times does the row swap happen
uint32_t swap_rows_times = 0;

#endif