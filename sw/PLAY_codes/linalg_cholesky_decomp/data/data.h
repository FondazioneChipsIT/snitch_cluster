#ifndef DATA_H
#define DATA_H

// Matrix elems (for now only a multiple of ncores, default 8)
#ifndef elems
#define elems 64
#endif

// TCDM pointers to our data
double *mat, *dst;

double local_sum[8];

// Cycle and performance metrics
uint64_t start_cycle[16], end_cycle[16], total_cycles[16]; // Increase the num if ncores>16
double flop_cycle[16];
// How many times does the row swap happen
uint32_t swap_rows_times = 0;

#endif