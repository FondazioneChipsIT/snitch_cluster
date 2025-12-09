#ifndef DATA_H
#define DATA_H


// Matrix rows (for now only a multiple of ncores, default 8)
#ifndef rows
#define rows 64
#endif
// Matrix columns (for now only a multiple of ncores, default 8)
#ifndef col
#define col 64
#endif

// TCDM pointers to our data
double *mat_a,*val;

// Cycle and performance metrics
uint64_t start_cycle[16], end_cycle[16], total_cycles[16]; // Increase the num if ncores>16
double flop_cycle[16];

#endif